//! `llmhost-bench`: prefill and decode throughput for llmhost, one JSON line per test.
//!
//! The tests mirror `llama-bench` so rows can be compared directly:
//! `pp<N>` is a prefill of N tokens, `tg<N>` is N single-token decode steps.
//!
//! ```text
//! cargo run --release -p llmhost-bench -- --threads 4,8 --prompt 512 --gen 128 --runs 5
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use llm_cpu::{argmax, with_threads};
use llm_engine::{read_npz, LayerTraces, Lfm2Model};
use serde_json::json;

struct Args {
    threads: Vec<usize>,
    prompt: usize,
    gen: usize,
    runs: usize,
    out: Option<PathBuf>,
    max_load: f64,
    weights: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        threads: vec![8],
        prompt: 512,
        gen: 128,
        runs: 5,
        out: None,
        max_load: 0.5,
        weights: "q4".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--threads" | "-t" => {
                a.threads = val()
                    .split(',')
                    .map(|s| s.trim().parse().expect("threads"))
                    .collect()
            }
            "--prompt" | "-p" => a.prompt = val().parse().expect("prompt"),
            "--gen" | "-n" => a.gen = val().parse().expect("gen"),
            "--runs" | "-r" => a.runs = val().parse().expect("runs"),
            "--out" | "-o" => a.out = Some(PathBuf::from(val())),
            "--max-load" => a.max_load = val().parse().expect("max-load"),
            "--weights" | "-w" => a.weights = val(),
            "-h" | "--help" => {
                eprintln!("usage: llmhost-bench [--threads 4,8] [--prompt 512] [--gen 128] [--runs 5] [--out results.jsonl] [--max-load 0.5] [--weights q4|f32]");
                std::process::exit(0);
            }
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn golden_ids(name: &str) -> Vec<u32> {
    let dir = std::env::var("LLMHOST_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("tests/golden/lfm2-700m"));
    let path = dir.join(format!("{name}.npz"));
    let dump = read_npz(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    dump.get("ids")
        .and_then(|a| a.as_i64())
        .expect("ids")
        .iter()
        .map(|&i| i as u32)
        .collect()
}

/// A prompt of exactly `n` real tokens: BOS, then the long reference text repeated.
fn prompt_of_len(n: usize) -> Vec<u32> {
    let src = golden_ids("long_en");
    let (bos, body) = (src[0], &src[1..]);
    let mut ids = vec![bos];
    while ids.len() < n {
        ids.extend(body.iter().take(n - ids.len()));
    }
    ids
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|x| x.parse().ok()))
        .unwrap_or(f64::NAN)
}

/// (busy, total) jiffies for the whole machine, from /proc/stat.
fn sys_jiffies() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let v: Vec<u64> = s
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let total: u64 = v.iter().sum();
    let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
    (total - idle, total)
}

/// utime + stime jiffies of this process (all threads).
fn own_jiffies() -> u64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    f.get(11).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0)
        + f.get(12).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0)
}

fn ncpu() -> f64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as f64)
        .unwrap_or(1.0)
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_default()
}

fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .current_dir(root())
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

struct Stats {
    mean: f64,
    stdev: f64,
    p50: f64,
    min: f64,
    max: f64,
}

fn stats(xs: &[f64]) -> Stats {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len() as f64;
    let mean = s.iter().sum::<f64>() / n;
    let var = if s.len() > 1 {
        s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    Stats {
        mean,
        stdev: var.sqrt(),
        p50: s[s.len() / 2],
        min: s[0],
        max: s[s.len() - 1],
    }
}

fn bench_prefill(model: &Lfm2Model, ids: &[u32], runs: usize) -> Vec<f64> {
    let _ = model.prefill(ids, None, ids.len()); // warmup
    (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            let _ = model.prefill(ids, None, ids.len());
            ids.len() as f64 / t0.elapsed().as_secs_f64()
        })
        .collect()
}

fn bench_decode(model: &Lfm2Model, gen: usize, runs: usize) -> Vec<f64> {
    let ids = golden_ids("short_en");
    let v = model.vocab();
    let one = |steps: usize| {
        let mut tr = LayerTraces::default();
        let mut cache = model.prefill(&ids, Some(&mut tr), ids.len() + steps + 1);
        let mut tok = argmax(&tr.logits[tr.logits.len() - v..]) as u32;
        let t0 = Instant::now();
        for _ in 0..steps {
            let logits = model.decode_one(tok, &mut cache);
            tok = argmax(&logits) as u32;
        }
        steps as f64 / t0.elapsed().as_secs_f64()
    };
    let _ = one(8); // warmup
    (0..runs).map(|_| one(gen)).collect()
}

fn main() {
    let a = parse_args();
    eprintln!("loading LFM2-700M ({})…", a.weights);
    let model = match a.weights.as_str() {
        "f32" => Lfm2Model::load_lfm2_700m(),
        "q4" => Lfm2Model::load_lfm2_700m_q4(),
        w => panic!("unknown --weights {w} (q4|f32)"),
    }
    .expect("LFM2-700M weights (LLMHOST_MODEL_DIR or Hub cache)");
    let prompt = prompt_of_len(a.prompt);

    let commit = git(&["rev-parse", "--short", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    let cpu = cpu_model();
    let mut out: Box<dyn std::io::Write> = match &a.out {
        Some(p) => Box::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("open --out"),
        ),
        None => Box::new(std::io::stdout()),
    };

    for &t in &a.threads {
        for test in [format!("pp{}", a.prompt), format!("tg{}", a.gen)] {
            let load_before = loadavg();
            let (b0, t0j) = sys_jiffies();
            let o0 = own_jiffies();
            let xs = with_threads(t, || {
                if test.starts_with("pp") {
                    bench_prefill(&model, &prompt, a.runs)
                } else {
                    bench_decode(&model, a.gen, a.runs)
                }
            });
            let load_after = loadavg();
            let (b1, t1j) = sys_jiffies();
            let o1 = own_jiffies();
            // CPUs kept busy by other processes during the test (0 = quiet machine).
            let other_cpus = ((b1 - b0) as f64 - (o1 - o0) as f64).max(0.0)
                / ((t1j - t0j) as f64).max(1.0)
                * ncpu();
            let s = stats(&xs);
            let row = json!({
                "engine": "llmhost", "model": "LFM2-700M", "weights": a.weights, "test": test,
                "threads": t, "runs": a.runs,
                "tok_s_mean": s.mean, "tok_s_stdev": s.stdev, "tok_s_p50": s.p50,
                "tok_s_min": s.min, "tok_s_max": s.max, "samples": xs,
                "load_before": load_before, "load_after": load_after,
                "other_cpus": other_cpus,
                "noisy": other_cpus > a.max_load,
                "commit": commit, "dirty": dirty, "cpu": cpu,
                "unix_time": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            });
            writeln!(out, "{row}").unwrap();
            eprintln!(
                "{test:>6} t={t}: {:.2} ± {:.2} tok/s (load {load_before:.2}→{load_after:.2})",
                s.mean, s.stdev
            );
        }
    }
}
