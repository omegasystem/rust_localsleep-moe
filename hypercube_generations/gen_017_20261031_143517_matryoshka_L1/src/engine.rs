//! engine.rs - Zero-GIL Asynchronous Dual-Stream Engine in Pure Rust
//! Employs native OS worker threads and crossbeam channels to completely decouple
//! foreground inference from background NREM sleep maintenance (Fisher SVD / pruning).

use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Maintenance job dispatched during expert NREM sleep
pub struct MaintenanceJob {
    pub expert_id: usize,
    pub job_fn: Box<dyn FnOnce() + Send + 'static>,
    pub on_complete_callback: Option<Box<dyn FnOnce() + Send + 'static>>,
}

/// Zero-GIL Asynchronous Dual-Stream Execution Engine
pub struct DualStreamEngine {
    tx: Sender<Option<MaintenanceJob>>,
    worker_handle: Mutex<Option<JoinHandle<()>>>,
}

impl DualStreamEngine {
    pub fn new() -> Arc<Self> {
        let (tx, rx): (Sender<Option<MaintenanceJob>>, Receiver<Option<MaintenanceJob>>) = unbounded();

        // Spawn a dedicated, high-priority native OS thread (Zero-GIL worker)
        let worker_handle = thread::Builder::new()
            .name("nrem-worker-stream".to_string())
            .spawn(move || {
                println!("🦀 \x1b[1;32m[DualStreamEngine]\x1b[0m Native NREM Sleep Worker Thread spawned (Zero-GIL active).");
                while let Ok(maybe_job) = rx.recv() {
                    match maybe_job {
                        Some(job) => {
                            let exp_id = job.expert_id;
                            // Execute heavy synaptic downscaling / pruning
                            (job.job_fn)();

                            // Trigger wake-up callback if provided
                            if let Some(cb) = job.on_complete_callback {
                                cb();
                            }
                            println!("🟢 \x1b[1;32m[DualStreamEngine]\x1b[0m Expert {} maintenance complete & synchronized.", exp_id);
                        }
                        None => {
                            // Shutdown signal
                            break;
                        }
                    }
                }
                println!("💤 \x1b[1;33m[DualStreamEngine]\x1b[0m Worker thread exiting cleanly.");
            })
            .expect("Failed to spawn native NREM worker thread");

        Arc::new(Self {
            tx,
            worker_handle: Mutex::new(Some(worker_handle)),
        })
    }

    /// Dispatches a background synaptic maintenance job without blocking the inference thread
    pub fn dispatch_maintenance<F, C>(&self, expert_id: usize, job_fn: F, on_complete: C)
    where
        F: FnOnce() + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        let job = MaintenanceJob {
            expert_id,
            job_fn: Box::new(job_fn),
            on_complete_callback: Some(Box::new(on_complete)),
        };

        if let Err(e) = self.tx.send(Some(job)) {
            eprintln!("❌ [DualStreamEngine Error] Failed to dispatch maintenance job: {}", e);
        }
    }

    /// Gracefully stops the worker thread
    pub fn shutdown(&self) {
        let _ = self.tx.send(None);
        let mut handle_lock = self.worker_handle.lock();
        if let Some(handle) = handle_lock.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DualStreamEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}
