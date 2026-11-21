//! Tests for queue uploads that go through the pooled staging memory
//! (`Queue::write_buffer` and `Queue::write_buffer_with` with small sizes).
//!
//! Small uploads are sub-allocated from shared, persistently mapped chunks
//! that are reused once the submission that read them has completed. These
//! tests check that every upload lands intact, including when many uploads
//! share a chunk, when chunks fill up and roll over, and when a chunk is
//! reused across submissions while earlier ones may still be in flight.

use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        STAGING_POOL_MANY_SMALL_WRITES,
        STAGING_POOL_WRITE_BUFFER_WITH,
        STAGING_POOL_CHUNK_ROLLOVER_ACROSS_SUBMISSIONS,
        STAGING_POOL_LARGE_AND_SMALL_WRITES,
    ]);
}

/// A byte pattern that differs per `seed` and per position.
fn pattern(seed: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            (i as u32)
                .wrapping_mul(31)
                .wrapping_add(seed.wrapping_mul(97)) as u8
        })
        .collect()
}

fn storage_buffer(ctx: &TestingContext, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

async fn read_back(ctx: &TestingContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &read, 0, buffer.size());
    ctx.queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    slice.get_mapped_range().unwrap().to_vec()
}

/// Hundreds of small writes of varied sizes and offsets into one buffer,
/// all in one submission: they share staging chunks.
#[gpu_test]
static STAGING_POOL_MANY_SMALL_WRITES: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        const SIZE: u64 = 256 * 1024;
        let buffer = storage_buffer(&ctx, SIZE);
        let mut expected = vec![0u8; SIZE as usize];
        let mut offset = 0usize;
        let mut seed = 0u32;
        // Sizes from 4 bytes to 4 KiB, all multiples of COPY_BUFFER_ALIGNMENT.
        while offset < SIZE as usize {
            let len = (4 * (1 + (seed as usize * 37) % 1024)).min(SIZE as usize - offset);
            let data = pattern(seed, len);
            ctx.queue.write_buffer(&buffer, offset as u64, &data);
            expected[offset..offset + len].copy_from_slice(&data);
            offset += len;
            seed += 1;
        }
        ctx.queue.submit([]);
        assert_eq!(read_back(&ctx, &buffer).await, expected);
    });

/// `write_buffer_with` (which creates a staging buffer handed to the caller)
/// interleaved with `write_buffer`, overlapping writes resolve in call order.
#[gpu_test]
static STAGING_POOL_WRITE_BUFFER_WITH: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        const SIZE: u64 = 64 * 1024;
        let buffer = storage_buffer(&ctx, SIZE);
        let mut expected = vec![0u8; SIZE as usize];
        for i in 0..200u32 {
            let offset = ((i as usize * 1236) % (SIZE as usize - 1024)) & !3;
            let len = 4 * (1 + (i as usize * 13) % 256);
            let data = pattern(i, len);
            if i % 2 == 0 {
                let mut view = ctx
                    .queue
                    .write_buffer_with(
                        &buffer,
                        offset as u64,
                        wgpu::BufferSize::new(len as u64).unwrap(),
                    )
                    .unwrap();
                view.copy_from_slice(&data);
            } else {
                ctx.queue.write_buffer(&buffer, offset as u64, &data);
            }
            expected[offset..offset + len].copy_from_slice(&data);
        }
        ctx.queue.submit([]);
        assert_eq!(read_back(&ctx, &buffer).await, expected);
    });

/// Many submissions, each uploading more small data than one chunk holds and
/// none waited on until the end: chunks roll over and get reused while earlier
/// submissions may still be executing. Every destination must see its own data.
#[gpu_test]
static STAGING_POOL_CHUNK_ROLLOVER_ACROSS_SUBMISSIONS: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        const WRITE: usize = 60 * 1024;
        const WRITES_PER_SUBMIT: usize = 80; // ~4.7 MiB per submission
        const SUBMITS: usize = 6;
        let buffers: Vec<wgpu::Buffer> = (0..SUBMITS)
            .map(|_| storage_buffer(&ctx, (WRITE * WRITES_PER_SUBMIT) as u64))
            .collect();
        for (s, buffer) in buffers.iter().enumerate() {
            for w in 0..WRITES_PER_SUBMIT {
                let data = pattern((s * WRITES_PER_SUBMIT + w) as u32, WRITE);
                ctx.queue.write_buffer(buffer, (w * WRITE) as u64, &data);
            }
            ctx.queue.submit([]);
        }
        for (s, buffer) in buffers.iter().enumerate() {
            let data = read_back(&ctx, buffer).await;
            for w in 0..WRITES_PER_SUBMIT {
                let expected = pattern((s * WRITES_PER_SUBMIT + w) as u32, WRITE);
                assert!(
                    data[w * WRITE..(w + 1) * WRITE] == expected[..],
                    "submission {s}, write {w} was overwritten"
                );
            }
        }
    });

/// Uploads above the pooling threshold keep a staging buffer of their own;
/// mixing them with pooled ones must not disturb either.
#[gpu_test]
static STAGING_POOL_LARGE_AND_SMALL_WRITES: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        const LARGE: usize = 3 * 1024 * 1024;
        let buffer = storage_buffer(&ctx, (2 * LARGE) as u64);
        let large_a = pattern(1, LARGE);
        let large_b = pattern(2, LARGE);
        ctx.queue.write_buffer(&buffer, 0, &large_a);
        let mut expected = large_a.clone();
        expected.extend_from_slice(&large_b);
        // Small writes land on top of the large one, then the second large one.
        for i in 0..64usize {
            let data = pattern(100 + i as u32, 256);
            ctx.queue.write_buffer(&buffer, (i * 4096) as u64, &data);
            expected[i * 4096..i * 4096 + 256].copy_from_slice(&data);
        }
        ctx.queue.write_buffer(&buffer, LARGE as u64, &large_b);
        ctx.queue.submit([]);
        assert_eq!(read_back(&ctx, &buffer).await, expected);
    });
