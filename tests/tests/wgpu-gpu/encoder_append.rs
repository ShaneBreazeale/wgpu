//! Tests for `CommandEncoder::append`, which moves the commands recorded on
//! one encoder to the end of another before either is finished.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use wgpu::util::DeviceExt as _;
use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        ENCODER_APPEND_RUNS_IN_ORDER,
        ENCODER_APPEND_MOVES_DEFERRED_ACTIONS,
        ENCODER_APPEND_REJECTS_LOCKED_ENCODERS,
        ENCODER_APPEND_EMPTY,
    ]);
}

fn filled(ctx: &TestingContext, byte: u8, size: usize) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &vec![byte; size],
            usage: wgpu::BufferUsages::COPY_SRC,
        })
}

fn target(ctx: &TestingContext, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn encoder(ctx: &TestingContext) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
}

async fn read(ctx: &TestingContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    slice.get_mapped_range().unwrap().to_vec()
}

/// The appended commands run after the destination's own, as one command buffer.
#[gpu_test]
static ENCODER_APPEND_RUNS_IN_ORDER: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let ones = filled(&ctx, 1, 64);
        let twos = filled(&ctx, 2, 64);
        let dst = target(&ctx, 128);

        let mut a = encoder(&ctx);
        a.copy_buffer_to_buffer(&ones, 0, &dst, 0, 64);
        let mut b = encoder(&ctx);
        // Overwrites what `a` wrote: only correct if it runs after `a`.
        b.copy_buffer_to_buffer(&twos, 0, &dst, 0, 64);
        b.copy_buffer_to_buffer(&ones, 0, &dst, 64, 64);
        a.append(b).map_err(|_| ()).expect("append failed");

        ctx.queue.submit(Some(a.finish()));
        let data = read(&ctx, &dst).await;
        assert!(data[..64].iter().all(|&v| v == 2));
        assert!(data[64..].iter().all(|&v| v == 1));
    });

/// Callbacks and buffer mappings requested on the appended encoder still happen
/// when the merged command buffer is submitted.
#[gpu_test]
static ENCODER_APPEND_MOVES_DEFERRED_ACTIONS: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let ones = filled(&ctx, 1, 64);
        let dst = target(&ctx, 64);
        let done = Arc::new(AtomicBool::new(false));
        let mapped = Arc::new(AtomicBool::new(false));

        let mut a = encoder(&ctx);
        let mut b = encoder(&ctx);
        b.copy_buffer_to_buffer(&ones, 0, &dst, 0, 64);
        let flag = done.clone();
        b.on_submitted_work_done(move || flag.store(true, Ordering::SeqCst));
        let flag = mapped.clone();
        b.map_buffer_on_submit(&dst, wgpu::MapMode::Read, .., move |result| {
            result.unwrap();
            flag.store(true, Ordering::SeqCst);
        });
        a.append(b).map_err(|_| ()).expect("append failed");

        ctx.queue.submit(Some(a.finish()));
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        assert!(done.load(Ordering::SeqCst));
        assert!(mapped.load(Ordering::SeqCst));
        assert!(dst
            .slice(..)
            .get_mapped_range()
            .unwrap()
            .iter()
            .all(|&v| v == 1));
    });

/// An encoder locked by an open pass can neither absorb nor be absorbed;
/// `append` hands the source back untouched and both stay usable.
#[gpu_test]
static ENCODER_APPEND_REJECTS_LOCKED_ENCODERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let ones = filled(&ctx, 1, 64);
        let dst = target(&ctx, 64);

        // Source locked.
        let mut a = encoder(&ctx);
        let mut b = encoder(&ctx);
        let pass = b
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
            .forget_lifetime();
        let b = a
            .append(b)
            .expect_err("append of a locked encoder succeeded");
        drop(pass);

        // Destination locked.
        let pass = a
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
            .forget_lifetime();
        let mut b = a
            .append(b)
            .expect_err("append into a locked encoder succeeded");
        drop(pass);

        // Both are still valid and keep their own commands.
        b.copy_buffer_to_buffer(&ones, 0, &dst, 0, 64);
        ctx.queue.submit([a.finish(), b.finish()]);
        assert!(read(&ctx, &dst).await.iter().all(|&v| v == 1));
    });

/// Appending an encoder that recorded nothing, or into one, is fine.
#[gpu_test]
static ENCODER_APPEND_EMPTY: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let ones = filled(&ctx, 1, 64);
        let dst = target(&ctx, 64);
        let mut a = encoder(&ctx);
        a.append(encoder(&ctx))
            .map_err(|_| ())
            .expect("append failed");
        let mut b = encoder(&ctx);
        b.copy_buffer_to_buffer(&ones, 0, &dst, 0, 64);
        a.append(b).map_err(|_| ()).expect("append failed");
        ctx.queue.submit(Some(a.finish()));
        assert!(read(&ctx, &dst).await.iter().all(|&v| v == 1));
    });
