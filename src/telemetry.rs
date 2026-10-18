//! telemetry.rs - High-Precision Microsecond Telemetry & Visualization Engine
//! Implements JSONL side-recording, ANSI terminal activity gauges, and Foxglove WebSocket payloads.

use crate::types::{ExpertTelemetry, TelemetryRecord};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct TelemetryLogger {
    log_file: PathBuf,
}

impl TelemetryLogger {
    pub fn new(session_dir: &Path, session_id: &str) -> Self {
        let log_file = session_dir.join(format!("session_rust_telemetry_{}.jsonl", session_id));
        Self { log_file }
    }

    /// Records high-precision microsecond telemetry record to disk
    pub fn record(&self, record: &TelemetryRecord) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.log_file) {
            if let Ok(line) = serde_json::to_string(record) {
                let _ = writeln!(file, "{}", line);
            }
        }
    }

    /// Renders a dynamic ANSI activity bar for the terminal: e.g. [████░░░░] 2.15/4.0
    pub fn render_gauge(val: f32, max_val: f32, length: usize) -> String {
        let ratio = (val / max_val).clamp(0.0, 1.0);
        let filled = (ratio * (length as f32)).round() as usize;
        let empty = length.saturating_sub(filled);

        let color = if ratio < 0.6 {
            "\x1b[1;32m" // Green
        } else if ratio < 0.9 {
            "\x1b[1;33m" // Yellow
        } else {
            "\x1b[1;31m" // Red Alert
        };

        let bar: String = "█".repeat(filled) + &"░".repeat(empty);
        format!("{}[{}]{}\x1b[0m {:4.2}/{:.1}", color, bar, color, val, max_val)
    }

    /// Prints a comprehensive real-time dashboard of all expert homeostatic states
    pub fn print_expert_dashboard(experts: &[ExpertTelemetry]) {
        print!("\r[Neural Dashboard] ");
        for exp in experts {
            let state_icon = match exp.state {
                0 => "\x1b[1;32m🟢\x1b[0m",
                1 => "\x1b[1;33m🟡\x1b[0m",
                2 => "\x1b[1;35m💤\x1b[0m",
                _ => "⚪",
            };
            print!(
                "E{}:{}(Fatigue:{:4.2}) ",
                exp.id, state_icon, exp.fatigue
            );
        }
        use std::io::stdout;
        let _ = stdout().flush();
    }
}
