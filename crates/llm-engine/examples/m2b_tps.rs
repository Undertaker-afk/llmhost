//! M2b throughput, profile, and Q4 quality variants for LFM2-700M.
//!
//! Decode: 128 incremental tokens after a short prompt and after a 512-token
//! prompt. Prefill is reported separately. Measurements skip a warmup pass
//! (weights are also pre-faulted at load). Thread counts: 4 and 8, plus a
//! 1-thread decode baseline. System load is recorded from `uptime`.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use llm_cpu::{amx_ready, argmax, detect_q4_kernel, set_act_quant, with_threads, ActQuant};
use llm_engine::{read_npz, DecodeScratch, LayerTraces, Lfm2Model, Q4LoadOpts};

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

fn load_ids(name: &str) -> Option<Vec<u32>> {
    let path = golden_dir().join(format!("{name}.npz"));
    if !path.is_file() {
        // Keep the note repo-relative; dumps are optional and not regenerated here.
        println!("note: tests/golden/lfm2-700m/{name}.npz is not present; skipping that prompt");
        return None;
    }
    let dump = read_npz(&path).unwrap_or_else(|e| panic!("read {name}.npz: {e}"));
    Some(
        dump.get("ids")
            .and_then(|a| a.as_i64())
            .expect("ids")
            .iter()
            .map(|&i| u32::try_from(i).expect("token id"))
            .collect(),
    )
}

fn uptime() -> String {
    Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "(uptime unavailable)".into())
}

/// One warmup decode of a few tokens, then time `steps` tokens. Returns
/// tok/s and an optional profile from the timed window.
fn decode_tps(
    model: &Lfm2Model,
    ids: &[u32],
    steps: usize,
    threads: usize,
    profile: bool,
) -> (f64, Option<llm_engine::DecodeProfile>) {
    with_threads(threads, || {
        let v = model.vocab();
        let warmup = 8.min(steps);
        let mut traces = LayerTraces::default();
        let mut cache = model.prefill(ids, Some(&mut traces), ids.len() + warmup + steps);
        let mut scratch = DecodeScratch::for_model(model, cache.kv_capacity());
        let mut tok = argmax(&traces.logits[traces.logits.len() - v..]) as u32;
        for _ in 0..warmup {
            let logits = model.decode_one_in(tok, &mut cache, &mut scratch);
            tok = argmax(logits) as u32;
        }
        scratch.collect_profile = profile;
        scratch.profile = llm_engine::DecodeProfile::default();
        let t0 = Instant::now();
        for _ in 0..steps {
            let logits = model.decode_one_in(tok, &mut cache, &mut scratch);
            tok = argmax(logits) as u32;
        }
        let dt = t0.elapsed().as_secs_f64();
        let prof = if profile {
            Some(scratch.profile.clone())
        } else {
            None
        };
        (steps as f64 / dt, prof)
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
    let mut scratch = DecodeScratch::for_model(model, cache.kv_capacity());
    for _ in 0..steps {
        toks.push(next);
        let row = model.decode_one_in(next, &mut cache, &mut scratch);
        next = argmax(row) as u32;
        if logits.len() < steps {
            logits.push(row.to_vec());
        }
    }
    (toks, logits)
}

fn forced_logits(model: &Lfm2Model, prompt: &[u32], tokens: &[u32]) -> Vec<Vec<f32>> {
    let v = model.vocab();
    let mut traces = LayerTraces::default();
    let mut cache = model.prefill(prompt, Some(&mut traces), prompt.len() + tokens.len());
    let mut current = traces.logits[traces.logits.len() - v..].to_vec();
    let mut out = Vec::with_capacity(tokens.len());
    let mut scratch = DecodeScratch::for_model(model, cache.kv_capacity());
    for &tok in tokens {
        out.push(current);
        current = model.decode_one_in(tok, &mut cache, &mut scratch).to_vec();
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

fn report_accuracy(label: &str, model: &Lfm2Model, fp32: &Lfm2Model, ids: &[u32], steps: usize) {
    let (fp_toks, fp_logits) = greedy_run(fp32, ids, steps);
    let (q_toks, _) = greedy_run(model, ids, steps);
    let agree = fp_toks
        .iter()
        .zip(q_toks.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let n_match = fp_toks
        .iter()
        .zip(q_toks.iter())
        .filter(|(a, b)| a == b)
        .count();
    let first_mismatch = fp_toks
        .iter()
        .zip(q_toks.iter())
        .enumerate()
        .find_map(|(i, (a, b))| if a != b { Some((i, *b, *a)) } else { None });
    let forced = forced_logits(model, ids, &fp_toks);
    let (mean, max) = mean_max_abs_logit_err(&forced, &fp_logits);
    print!("  {label}: free-greedy {n_match}/{steps} tokens agree");
    if let Some((step, got, exp)) = first_mismatch {
        print!(" (first mismatch at step {step}: q={got}, fp32={exp}, prefix {agree})");
    }
    println!("; teacher-forced mean |dlogit|={mean:.4e}, max={max:.4e}");
}

fn main() {
    println!("system load (start): {}", uptime());
    println!("Q4 kernel: {:?}", detect_q4_kernel());
    println!("AMX-INT8 ready: {}", amx_ready());

    println!("loading LFM2-700M Q4_32 (linears + tied head; embeddings/norms fp32)…");
    let t0 = Instant::now();
    let q4 = Lfm2Model::load_lfm2_700m_q4().expect("Q4 load");
    println!(
        "loaded Q4 in {:.1}s  format={:?}  (weights pre-faulted)",
        t0.elapsed().as_secs_f64(),
        q4.format
    );

    let short = load_ids("short_en").expect("short_en.npz is required for M2b measurements");
    let long = load_ids("long_en");
    let code = load_ids("code_py");
    let decode_steps = 128;

    // 512-token prompt: repeat long_en when the dump is present, otherwise
    // repeat short_en (token identity does not change arithmetic intensity).
    let prompt_src = long.as_deref().unwrap_or(short.as_slice());
    let mut prompt512 = Vec::with_capacity(512);
    while prompt512.len() < 512 {
        prompt512.extend_from_slice(prompt_src);
    }
    prompt512.truncate(512);
    if long.is_none() {
        println!("prefill-512 / decode@512 use a repeated short_en prompt (long_en.npz missing)");
    }

    println!();
    println!(
        "profile: short_en prefill T={} then {decode_steps} incremental tokens, 8 threads",
        short.len()
    );
    let (tps, prof) = decode_tps(&q4, &short, decode_steps, 8, true);
    println!(
        "  8-thread decode (profiled): {tps:.2} tok/s    load: {}",
        uptime()
    );
    if let Some(p) = prof {
        let total_ms = (p.gemv_ns + p.attn_ns + p.conv_ns + p.norm_ns + p.other_ns) as f64
            / 1e6
            / p.tokens.max(1) as f64;
        println!("  per-token {:.3} ms  ({} tokens):", total_ms, p.tokens);
        for (name, ns, frac) in p.fractions() {
            let us = ns as f64 / 1e3 / p.tokens.max(1) as f64;
            println!("    {name:5}  {us:8.1} µs/tok  ({:5.1}%)", frac * 100.0);
        }
    }

    println!();
    println!(
        "decode: short_en prefill T={} then {decode_steps} incremental tokens (Q4_32, warmup skipped)",
        short.len()
    );
    for th in [1usize, 4, 8] {
        let (tps, _) = decode_tps(&q4, &short, decode_steps, th, false);
        println!("  {th}-thread decode: {tps:.2} tok/s    load: {}", uptime());
    }

    if let Some(ref long) = long {
        println!("prefill: long_en T={} (Q4_32)", long.len());
        for th in [1usize, 4, 8] {
            let (tps, dt) = prefill_tps(&q4, long, th);
            println!(
                "  {th}-thread prefill: {tps:.2} tok/s  ({dt:.3}s)    load: {}",
                uptime()
            );
        }
    }

    println!("prefill: 512-token prompt (Q4_32)");
    for th in [4usize, 8] {
        let (tps, dt) = prefill_tps(&q4, &prompt512, th);
        println!(
            "  {th}-thread prefill-512: {tps:.2} tok/s  ({dt:.3}s)    load: {}",
            uptime()
        );
    }
    println!(
        "decode: 512-token prompt then {decode_steps} incremental tokens (Q4_32, warmup skipped)"
    );
    for th in [4usize, 8] {
        let (tps, _) = decode_tps(&q4, &prompt512, decode_steps, th, false);
        println!(
            "  {th}-thread decode@512: {tps:.2} tok/s    load: {}",
            uptime()
        );
    }

    if std::env::var("LLMHOST_M2B_THROUGHPUT_ONLY").is_ok() {
        println!("skipping accuracy (LLMHOST_M2B_THROUGHPUT_ONLY)");
        println!("system load (end): {}", uptime());
        return;
    }

    println!();
    println!("accuracy vs fp32 greedy (128 tokens)");
    println!("loading LFM2-700M fp32…");
    let t0 = Instant::now();
    let fp32 = Lfm2Model::load_lfm2_700m().expect("fp32 load");
    println!("loaded fp32 in {:.1}s", t0.elapsed().as_secs_f64());

    println!("  variant: Q4 body + Q4 head, per-block act quant (default)");
    let acc_sets: Vec<(&str, &[u32])> = {
        let mut v = vec![("short_en", short.as_slice())];
        if let Some(ref c) = code {
            v.push(("code_py", c.as_slice()));
        }
        v
    };
    for (name, ids) in &acc_sets {
        report_accuracy(
            &format!("Q4/Q4 block {name}"),
            &q4,
            &fp32,
            ids,
            decode_steps,
        );
    }

    println!("  variant: Q4 body + Q4 head, per-row act quant");
    set_act_quant(ActQuant::PerRow);
    for (name, ids) in &acc_sets {
        report_accuracy(&format!("Q4/Q4 row {name}"), &q4, &fp32, ids, decode_steps);
    }
    set_act_quant(ActQuant::PerBlock);

    println!("loading LFM2-700M Q4 body + Q8_0 head…");
    let t0 = Instant::now();
    let q4q8 = Lfm2Model::load_lfm2_700m_q4_opts(Q4LoadOpts { q8_head: true }).expect("Q4+Q8 load");
    println!("loaded Q4+Q8 head in {:.1}s", t0.elapsed().as_secs_f64());
    println!("  variant: Q4 body + Q8 head, per-block act quant");
    for (name, ids) in &acc_sets {
        report_accuracy(
            &format!("Q4/Q8 block {name}"),
            &q4q8,
            &fp32,
            ids,
            decode_steps,
        );
    }

    println!("system load (end): {}", uptime());
}
