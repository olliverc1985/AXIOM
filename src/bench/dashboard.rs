//! Live HTML dashboard served from a raw TCP listener — zero external dependencies.
//!
//! Routes:
//! - `GET /`      → embedded HTML dashboard
//! - `GET /state` → current bench state as JSON (polled by the browser)

use serde::Serialize;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const INDEX_HTML: &str = include_str!("dashboard.html");

/// Snapshot of a single iteration for the trace log.
#[derive(Clone, Serialize)]
pub struct TraceSnapshot {
    pub iteration: usize,
    pub tier: String,
    pub trace: Vec<String>,
    pub confidence: f32,
    pub cost: f32,
    pub cached: bool,
}

/// Shared state between the bench loop and the dashboard server.
#[derive(Clone, Serialize)]
pub struct DashboardState {
    pub iteration: usize,
    pub total_iterations: usize,
    pub surface_count: usize,
    pub reasoning_count: usize,
    pub deep_count: usize,
    pub cache_hit_rate: f32,
    pub cache_size: usize,
    pub avg_compute_cost: f32,
    pub avg_confidence: f32,
    pub elapsed_ms: u64,
    pub recent_traces: VecDeque<TraceSnapshot>,
    pub confidence_history: VecDeque<f32>,
    pub done: bool,
    /// Traversal direction counts.
    pub forward_steps: usize,
    pub lateral_steps: usize,
    pub feedback_steps: usize,
    pub temporal_steps: usize,
    /// Lateral resolution stats.
    pub lateral_prevented: usize,
    /// Total feedback signals emitted.
    pub feedback_signals: usize,
}

impl DashboardState {
    pub fn new(total_iterations: usize) -> Self {
        Self {
            iteration: 0,
            total_iterations,
            surface_count: 0,
            reasoning_count: 0,
            deep_count: 0,
            cache_hit_rate: 0.0,
            cache_size: 0,
            avg_compute_cost: 0.0,
            avg_confidence: 0.0,
            elapsed_ms: 0,
            recent_traces: VecDeque::with_capacity(30),
            confidence_history: VecDeque::with_capacity(200),
            done: false,
            forward_steps: 0,
            lateral_steps: 0,
            feedback_steps: 0,
            temporal_steps: 0,
            lateral_prevented: 0,
            feedback_signals: 0,
        }
    }

    pub fn push_trace(&mut self, snap: TraceSnapshot) {
        self.confidence_history.push_back(snap.confidence);
        if self.confidence_history.len() > 200 {
            self.confidence_history.pop_front();
        }
        self.recent_traces.push_back(snap);
        if self.recent_traces.len() > 25 {
            self.recent_traces.pop_front();
        }
    }
}

/// Parse the request line from a TCP stream.
/// Reads headers fully so the connection is in a clean state for the response.
fn parse_request(stream: &mut TcpStream) -> Option<String> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    // Drain remaining headers (read until empty line)
    let mut header = String::new();
    loop {
        header.clear();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) => {
                if header.trim().is_empty() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    Some(request_line)
}

/// Send a full HTTP response with proper headers.
fn send_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         Cache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\n\
         \r\n",
        status,
        content_type,
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

/// Handle a single HTTP connection.
fn handle_connection(mut stream: TcpStream, state: &Arc<Mutex<DashboardState>>) {
    let request_line = match parse_request(&mut stream) {
        Some(r) => r,
        None => return,
    };

    // Route: "GET / HTTP/1.1" or "GET /state HTTP/1.1"
    let path = request_line.split_whitespace().nth(1).unwrap_or("");

    match path {
        "/" => {
            send_response(
                &mut stream,
                "200 OK",
                "text/html; charset=utf-8",
                INDEX_HTML.as_bytes(),
            );
        }
        "/state" => {
            let s = state.lock().unwrap();
            let json = serde_json::to_string(&*s).unwrap_or_default();
            drop(s);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes());
        }
        // Favicon — Safari always requests this
        "/favicon.ico" => {
            send_response(&mut stream, "204 No Content", "text/plain", b"");
        }
        _ => {
            send_response(&mut stream, "404 Not Found", "text/plain", b"Not Found");
        }
    }
}

/// Start the dashboard HTTP server on the given port.
///
/// Spawns a background thread with a blocking accept loop.
/// Each connection is handled in its own short-lived thread.
pub fn start_server(state: Arc<Mutex<DashboardState>>, port: u16) {
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .unwrap_or_else(|e| panic!("Failed to bind to port {}: {}", port, e));

    thread::spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let state = Arc::clone(&state);
                    thread::spawn(move || {
                        handle_connection(stream, &state);
                    });
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
    });
}
