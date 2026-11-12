//! Informational LFM2-700M Q4_32 throughput and a Q4-vs-fp32 accuracy check.
//!
//! Decode: `short_en` prefill then 128 generated tokens, plus a 512-token
//! prompt then 128 generated tokens. Prefill is reported separately.
//! Thread counts: 1, 4, and 8. System load is recorded from `uptime`.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use llm_cpu::{argmax, detect_q4_kernel, with_threads};
use llm_engine::{read_npz, LayerTraces, Lfm2Model};

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is in crates/<name>")
        .to_path_buf()
}

fn golden_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLMHOST_GOLDEN_DIR") {
        return PathBuf::from(dir);
    }
    workspace_root().join("tests/golden/lfm2-700m")
}

fn load_ids(name: &str) -> Vec<u32> {
    let path = golden_dir().join(format!("{name}.npz"));
    let dump = read_npz(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    dump.get("ids")
        .and_then(|a| a.as_i64())
        .expect("ids")
        .iter()
        .map(|&i| u32::try_from(i).expect("token id"))
        .collect()
}

fn uptime() -> String {
    Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "(uptime unavailable)".into())
}

fn decode_tps(model: &Lfm2Model, ids: &[u32], steps: usize, threads: usize) -> f64 {
    with_threads(threads, || {
        let v = model.vocab();
        let mut traces = LayerTraces::default();
        let mut cache = model.prefill(ids, Some(&mut traces), ids.len() + steps);
        let tok = argmax(&traces.logits[traces.logits.len() - v..]) as u32;
        let _ = model.decode_one(tok, &mut cache);

        let mut traces = LayerTraces::default();
        let mut cache = model.prefill(ids, Some(&mut traces), ids.len() + steps);
        let mut tok = argmax(&traces.logits[traces.logits.len() - v..]) as u32;
        let t0 = Instant::now();
        for _ in 0..steps {
            let logits = model.decode_one(tok, &mut cache);
            tok = argmax(&logits) as u32;
        }
        let dt = t0.elapsed().as_secs_f64();
        steps as f64 / dt
    })
}

fn prefill_tps(model: &Lfm2Model, ids: &[u32], threads: usize) -> (f64, f64) {
    with_threads(threads, || {
        let _ = model.prefill(ids, None, ids.len());
        let t0 = Instant::now();
        let _ = model.prefill(ids, None, ids.len());
        let dt = t0.elapsed().as_secs_f64();
        (ids.len() as f64 / dt, dt)
    })
}

fn greedy_run(model: &Lfm2Model, ids: &[u32], steps: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let v = model.vocab();
    let mut traces = LayerTraces::default();
    let mut cache = model.prefill(ids, Some(&mut traces), ids.len() + steps);
    let last = traces.logits[traces.logits.len() - v..].to_vec();
    let mut next = argmax(&last) as u32;
    let mut toks = Vec::with_capacity(steps);
    let mut logits = Vec::with_capacity(steps);
    logits.push(last);
    for _ in 0..steps {
        toks.push(next);
        let row = model.decode_one(next, &mut cache);
        next = argmax(&row) as u32;
        if logits.len() < steps {
            logits.push(row);
        }
    }
    // We stored the pre-step logits for each of the `steps` tokens.
    (toks, logits)
}

/// Teacher-force `tokens` and collect the logits that produced each one
/// (plus we ignore the trailing unused decode after the last token).
fn forced_logits(model: &Lfm2Model, prompt: &[u32], tokens: &[u32]) -> Vec<Vec<f32>> {
    let v = model.vocab();
    let mut traces = LayerTraces::default();
    let mut cache = model.prefill(prompt, Some(&mut traces), prompt.len() + tokens.len());
    let mut current = traces.logits[traces.logits.len() - v..].to_vec();
    let mut out = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        out.push(current);
        current = model.decode_one(tok, &mut cache);
    }
    out
}

fn mean_max_abs_logit_err(got: &[Vec<f32>], exp: &[Vec<f32>]) -> (f64, f32) {
    assert_eq!(got.len(), exp.len());
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut max = 0.0f32;
    for (g, e) in got.iter().zip(exp.iter()) {
        assert_eq!(g.len(), e.len());
        for (&a, &b) in g.iter().zip(e.iter()) {
            let d = (a - b).abs();
            sum += f64::from(d);
            count += 1;
            if d > max {
                max = d;
            }
        }
    }
    (sum / count as f64, max)
}

fn main() {
    println!("system load: {}", uptime());
    println!("Q4 kernel: {:?}", detect_q4_kernel());

    println!("loading LFM2-700M Q4_32 (linears + tied head; embeddings/norms fp32)…");
    let t0 = Instant::now();
    let q4 = Lfm2Model::load_lfm2_700m_q4().expect("Q4 load");
    println!(
        "loaded Q4 in {:.1}s  format={:?}",
        t0.elapsed().as_secs_f64(),
        q4.format
    );

    let short = load_ids("short_en");
    let long = load_ids("long_en");
    let code = load_ids("code_py");
    let decode_steps = 128;

    // 512-token prompt: repeat long_en and trim.
    let mut prompt512 = Vec::with_capacity(512);
    while prompt512.len() < 512 {
        prompt512.extend_from_slice(&long);
    }
    prompt512.truncate(512);

    println!();
    println!(
        "decode: short_en prefill T={} then {decode_steps} incremental tokens (Q4_32)",
        short.len()
    );
    for th in [1usize, 4, 8] {
        let tps = decode_tps(&q4, &short, decode_steps, th);
        println!("  {th}-thread decode: {tps:.2} tok/s    load: {}", uptime());
    }

    println!("prefill: long_en T={} (Q4_32)", long.len());
    for th in [1usize, 4, 8] {
        let (tps, dt) = prefill_tps(&q4, &long, th);
        println!(
            "  {th}-thread prefill: {tps:.2} tok/s  ({dt:.3}s)    load: {}",
            uptime()
        );
    }

    println!("prefill: 512-token prompt (Q4_32)");
    for th in [4usize, 8] {
        let (tps, dt) = prefill_tps(&q4, &prompt512, th);
        println!(
            "  {th}-thread prefill-512: {tps:.2} tok/s  ({dt:.3}s)    load: {}",
            uptime()
        );
    }
    println!("decode: 512-token prompt then {decode_steps} incremental tokens (Q4_32)");
    for th in [4usize, 8] {
        let tps = decode_tps(&q4, &prompt512, decode_steps, th);
        println!(
            "  {th}-thread decode@512: {tps:.2} tok/s    load: {}",
            uptime()
        );
    }

    println!();
    println!("accuracy: Q4 greedy vs fp32 greedy (128 tokens)");
    println!("loading LFM2-700M fp32…");
    let t0 = Instant::now();
    let fp32 = Lfm2Model::load_lfm2_700m().expect("fp32 load");
    println!("loaded fp32 in {:.1}s", t0.elapsed().as_secs_f64());

    for (name, ids) in [("short_en", short.as_slice()), ("code_py", code.as_slice())] {
        let (fp_toks, fp_logits) = greedy_run(&fp32, ids, decode_steps);
        let (q4_toks, _) = greedy_run(&q4, ids, decode_steps);
        let agree = fp_toks
            .iter()
            .zip(q4_toks.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let n_match = fp_toks
            .iter()
            .zip(q4_toks.iter())
            .filter(|(a, b)| a == b)
            .count();
        let first_mismatch = fp_toks
            .iter()
            .zip(q4_toks.iter())
            .enumerate()
            .find_map(|(i, (a, b))| if a != b { Some((i, *b, *a)) } else { None });
        let forced = forced_logits(&q4, ids, &fp_toks);
        let (mean, max) = mean_max_abs_logit_err(&forced, &fp_logits);
        print!("  {name}: free-greedy {n_match}/{decode_steps} tokens agree");
        if let Some((step, got, exp)) = first_mismatch {
            print!(" (first mismatch at step {step}: Q4={got}, fp32={exp}, prefix {agree})");
        }
        println!("; teacher-forced mean |dlogit|={mean:.4e}, max={max:.4e}");
    }

    println!("system load (end): {}", uptime());
}
