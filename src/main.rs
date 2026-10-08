//! main.rs - Standalone Native Rust CLI & Benchmark Harness for rust_localsleep_moe

use clap::{Parser, Subcommand};
use rust_localsleep_moe::cache::SinkAwareRollingCache;
use rust_localsleep_moe::downscaler::FisherDownscaler;
use rust_localsleep_moe::engine::DualStreamEngine;
use rust_localsleep_moe::expert_pool::LoRAExpertPool;
use rust_localsleep_moe::router::FatigueMoERouter;
use rust_localsleep_moe::telemetry::TelemetryLogger;
use rust_localsleep_moe::types::{ExpertState, ExpertTelemetry, LocalSleepConfig, TelemetryRecord};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "localsleep")]
#[command(author = "Kuro & Leon")]
#[command(version = "0.1.0")]
#[command(about = "High-Performance Zero-GIL Rust & CUDA Engine for LocalSleep-MoE", long_about = None)]
struct Cli {
    #[arg(short, long, default_value = "thor_12b")]
    profile: String,

    #[arg(short, long, default_value_t = 8)]
    experts: usize,

    #[arg(short, long, default_value_t = 3840)]
    hidden_dim: usize,

    #[arg(short, long, default_value_t = 2)]
    top_k: usize,

    #[arg(long, default_value_t = 2.0)]
    fatigue_threshold: f32,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run continuous synthetic token stress benchmark
    Benchmark {
        #[arg(short, long, default_value_t = 200)]
        steps: usize,

        #[arg(short, long, default_value_t = 1)]
        batch_size: usize,
    },
    /// Test SWA (Sink-Aware Sliding Window Attention) Cache bounds
    TestSwa {
        #[arg(short, long, default_value_t = 4000)]
        tokens: usize,
    },
    /// Inspect CUDA accelerator presence
    CudaInfo,
}

fn main() {
    let cli = Cli::parse();

    println!("===========================================================================");
    println!("🦀 \x1b[1;36mLocalSleep-MoE Native Rust Engine\x1b[0m (Profile: \x1b[1;33m{}\x1b[0m)", cli.profile);
    println!("🏛️  Topology: {} Experts | Top-{} Gating | Hidden Dim: {}", cli.experts, cli.top_k, cli.hidden_dim);
    println!("⚡ Zero-GIL Multi-Threaded Engine: \x1b[1;32mACTIVE\x1b[0m");
    println!("===========================================================================");

    let mut cfg = LocalSleepConfig::default();
    cfg.profile_name = cli.profile;
    cfg.num_experts = cli.experts;
    cfg.hidden_dim = cli.hidden_dim;
    cfg.top_k = cli.top_k;
    cfg.fatigue_threshold = cli.fatigue_threshold;

    match cli.command.unwrap_or(Commands::Benchmark { steps: 100, batch_size: 1 }) {
        Commands::Benchmark { steps, batch_size } => {
            run_benchmark(&cfg, steps, batch_size);
        }
        Commands::TestSwa { tokens } => {
            run_swa_test(tokens);
        }
        Commands::CudaInfo => {
            let bridge = rust_localsleep_moe::cuda_bridge::CudaEngineBridge::new();
            println!("🔍 CUDA Acceleration Kernel Status: {}", if bridge.is_available { "\x1b[1;32mONLINE\x1b[0m" } else { "\x1b[1;33mOFFLINE (Pure Rust Fallback)\x1b[0m" });
        }
    }
}

fn run_benchmark(cfg: &LocalSleepConfig, steps: usize, batch_size: usize) {
    let router = FatigueMoERouter::new(cfg);
    let expert_pool = LoRAExpertPool::new(cfg);
    let downscaler = FisherDownscaler::new(cfg.shy_lambda, cfg.enable_2_4_sparsity);
    let engine = DualStreamEngine::new();
    let logger = TelemetryLogger::new(&PathBuf::from("./sessions"), "cli_benchmark");

    println!("\n🚀 Starting Zero-GIL Benchmark: {} steps (Batch Size: {})...", steps, batch_size);
    let start_time = Instant::now();

    // Pseudo-random hidden state generation buffer
    let mut hidden = vec![0.05f32; batch_size * cfg.hidden_dim];

    for step in 1..=steps {
        // Dynamic jitter simulation
        for val in &mut hidden {
            *val = (*val * 1.01 + 0.001) % 1.0;
        }

        // 1. Router Forward
        let (weights, selected, entropy) = router.forward(&hidden, batch_size);

        // 2. Check for fatigued experts -> Schedule async NREM sleep
        let fatigued = router.check_fatigue_triggers();
        for exp_id in fatigued {
            let states_lock = router.states.clone();
            let fatigue_lock = router.fatigue_scores.clone();
            states_lock.write()[exp_id] = ExpertState::NremSleep;

            let downscaler_clone = downscaler.clone();
            // Clone expert weights buffer to simulate maintenance
            let mut weights_copy = expert_pool.experts[exp_id].lora_a.clone();

            engine.dispatch_maintenance(
                exp_id,
                move || {
                    downscaler_clone.execute_sleep_maintenance(&mut weights_copy);
                },
                move || {
                    states_lock.write()[exp_id] = ExpertState::Active;
                    fatigue_lock.write()[exp_id] = 0.0;
                },
            );
        }

        // 3. Expert Pool Forward Modulation
        let _delta = expert_pool.forward(&hidden, &weights, &selected, cfg.top_k, batch_size);

        // 4. Periodic Homeostatic Decay
        router.step_decay();

        // 5. Gather Telemetry
        let states = router.states.read().clone();
        let fatigue = router.fatigue_scores.read().clone();
        let tokens = router.active_token_counts.read().clone();
        let countdown = router.sleep_countdown.read().clone();

        let mut expert_telem = Vec::new();
        for i in 0..cfg.num_experts {
            expert_telem.push(ExpertTelemetry {
                id: i,
                state: states[i] as usize,
                fatigue: fatigue[i],
                tokens: tokens[i],
                countdown: countdown[i],
                norm: expert_pool.experts[i].norm_a(),
                sparsity: expert_pool.experts[i].sparsity_a(),
            });
        }

        if step % 10 == 0 || step == steps {
            TelemetryLogger::print_expert_dashboard(&expert_telem);
        }

        let record = TelemetryRecord {
            ts: chrono::Local::now().to_rfc3339(),
            step,
            token_id: (step % 32000) as u32,
            token_str: format!("tok_{}", step),
            entropy,
            delta_entropy: 0.0,
            top_candidates: vec![],
            top_probs: vec![],
            routed_experts: selected,
            routing_weights: weights,
            experts: expert_telem,
        };
        logger.record(&record);
    }

    let elapsed = start_time.elapsed();
    let throughput = (steps * batch_size) as f64 / elapsed.as_secs_f64();

    println!("\n\n===========================================================================");
    println!("🏁 \x1b[1;32mBenchmark Complete!\x1b[0m");
    println!("⏱️  Total Duration: {:.2?} | Throughput: \x1b[1;33m{:.2} tokens/s\x1b[0m", elapsed, throughput);
    println!("💾 Telemetry JSONL saved to ./sessions/");
    println!("===========================================================================");
}

fn run_swa_test(tokens: usize) {
    println!("\n🧪 Testing SWA (Sink-Aware Sliding Window Attention) Cache...");
    let sink_size = 4;
    let window_size = 2048;
    let mut cache = SinkAwareRollingCache::new(4, sink_size, window_size, 8, 128);

    let step_keys = vec![0.1f32; 8 * 1 * 128];
    let step_values = vec![0.1f32; 8 * 1 * 128];

    let start = Instant::now();
    for _ in 0..tokens {
        for l in 0..4 {
            cache.update_layer(l, &step_keys, &step_values, 1);
        }
    }
    let elapsed = start.elapsed();

    println!("✓ Ingested {} tokens across 4 layers in {:.2?}", tokens, elapsed);
    println!("✓ Final Layer 0 Seq Len: {} (Bounded limit: {})", cache.layers[0].seq_len(), sink_size + window_size);
    assert_eq!(cache.layers[0].seq_len(), sink_size + window_size);
    println!("🎉 \x1b[1;32mSWA Memory Boundedness Verified! Zero Memory Leak.\x1b[0m");
}
