//! Milestone 1 gate: LFM2-700M fp32 forward vs Hugging Face reference dumps.
//!
//! Per-layer residual uses atol+rtol: `|a-b| <= 1e-4 + 1e-5*|b|`. Final
//! logits stay a pure absolute gate of 1e-3. Greedy 128-token incremental
//! decode must match `greedy_ids` exactly on short_en, code_py, and
//! chat_basic. When greedy mismatches, a teacher-forced pass (feeding the
//! reference tokens) compares logits at every step so the first diverging
//! decode step is reported. Skips cleanly when dumps or weights are absent.

use llm_cpu::{argmax, with_threads};
use llm_engine::{read_npz, LayerTraces, Lfm2Model, NpyArray};
use std::path::{Path, PathBuf};
use std::time::Instant;

const LAYER_ATOL: f32 = 1e-4;
const LAYER_RTOL: f32 = 1e-5;
const LOGIT_TOL: f32 = 1e-3;
const PROMPTS: &[&str] = &[
    "short_en",
    "code_py",
    "multilingual",
    "long_en",
    "chat_basic",
];
const GREEDY_PROMPTS: &[&str] = &["short_en", "code_py", "chat_basic"];
// The reference dumps called `generate(..., pad_token_id=eos_token_id)` with
// eos=7, so prompt positions equal to 7 are padding. Layer traces were
// captured from an unmasked forward and still use `prefill` (no pad id).
const GENERATE_PAD_TOKEN_ID: u32 = 7;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is in crates/<name>")
        .to_path_buf()
}

fn golden_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LLMHOST_GOLDEN_DIR") {
        return Some(PathBuf::from(dir));
    }
    let default = workspace_root().join("tests/golden/lfm2-700m");
    if default.join("manifest.json").is_file() {
        Some(default)
    } else {
        None
    }
}

fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    assert_eq!(got.len(), exp.len(), "length mismatch");
    got.iter()
        .zip(exp.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn layer_allowed(exp: f32) -> f32 {
    LAYER_ATOL + LAYER_RTOL * exp.abs()
}

fn layer_violation(got: &[f32], exp: &[f32]) -> (f32, usize, Option<(usize, f32, f32)>) {
    assert_eq!(got.len(), exp.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    let mut n_over = 0usize;
    let mut first = None;
    for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > layer_allowed(e) {
            n_over += 1;
            if first.is_none() {
                first = Some((i, g, e));
            }
        }
    }
    (max_abs, n_over, first)
}

fn peak_err(got: &[f32], exp: &[f32]) -> (usize, f32, f32) {
    assert_eq!(got.len(), exp.len(), "length mismatch");
    let mut best = 0usize;
    let mut best_err = -1.0f32;
    for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
        let d = (g - e).abs();
        if d > best_err {
            best_err = d;
            best = i;
        }
    }
    (best, got[best], exp[best])
}

fn require_f32<'a>(map: &'a std::collections::HashMap<String, NpyArray>, key: &str) -> &'a [f32] {
    map.get(key)
        .and_then(|a| a.as_f32())
        .unwrap_or_else(|| panic!("missing f32 array {key}"))
}

fn require_i64<'a>(map: &'a std::collections::HashMap<String, NpyArray>, key: &str) -> &'a [i64] {
    map.get(key)
        .and_then(|a| a.as_i64())
        .unwrap_or_else(|| panic!("missing i64 array {key}"))
}

struct PromptReport {
    name: String,
    tokens: usize,
    embed_err: f32,
    layer_err: Vec<f32>,
    final_norm_err: f32,
    logits_err: f32,
    first_layer_over: Option<(usize, f32)>,
    layer_over_count: Vec<usize>,
    greedy_ok: Option<bool>,
    greedy_first_mismatch: Option<(usize, u32, u32)>,
    /// Teacher-forced: first step whose logits exceed LOGIT_TOL, if any.
    forced_first_logit_div: Option<(usize, f32)>,
    /// Teacher-forced: first step whose argmax != reference token, if any.
    forced_first_argmax_div: Option<(usize, u32, u32)>,
    forced_max_logit_err: Option<f32>,
}

impl PromptReport {
    fn layers_ok(&self) -> bool {
        self.layer_over_count.iter().all(|n| *n == 0)
    }
    fn logits_ok(&self) -> bool {
        self.logits_err <= LOGIT_TOL
    }
    fn force_feed_ok(&self) -> bool {
        self.forced_first_logit_div.is_none()
    }
}

fn eval_prompt(model: &Lfm2Model, dir: &Path, name: &str, do_greedy: bool) -> PromptReport {
    let path = dir.join(format!("{name}.npz"));
    let dump = read_npz(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let ids: Vec<u32> = require_i64(&dump, "ids")
        .iter()
        .map(|&i| u32::try_from(i).expect("token id"))
        .collect();
    let reserve = ids.len() + if do_greedy { 128 } else { 0 };

    let mut traces = LayerTraces::default();
    let _ = model.prefill(&ids, Some(&mut traces), reserve);

    let embed_err = max_abs_err(&traces.embed, require_f32(&dump, "embed"));
    let mut layer_err = Vec::with_capacity(16);
    let mut layer_over_count = Vec::with_capacity(16);
    let mut first_layer_over = None;
    for i in 0..16 {
        let key = format!("layer_{i:02}");
        let exp = require_f32(&dump, &key);
        let (err, n_over, first) = layer_violation(&traces.layers[i], exp);
        if n_over > 0 && first_layer_over.is_none() {
            first_layer_over = Some((i, err));
            let hidden = model.hidden();
            let (idx, got, want) = first.unwrap_or_else(|| peak_err(&traces.layers[i], exp));
            let tok = idx / hidden;
            let ch = idx % hidden;
            eprintln!(
                "  {name} {key} first atol+rtol violation at token {tok} channel {ch}: got={got} expected={want} abs={:.6e} allowed={:.6e} (token_id={})",
                (got - want).abs(),
                layer_allowed(want),
                ids[tok]
            );
            eprintln!("  {name} {key}: {n_over} elements over atol+rtol, max abs {err:.6e}");
        }
        layer_err.push(err);
        layer_over_count.push(n_over);
    }
    let final_norm_err = max_abs_err(&traces.final_norm, require_f32(&dump, "final_norm"));
    let logits_err = max_abs_err(&traces.logits, require_f32(&dump, "logits"));

    let mut greedy_ok = None;
    let mut greedy_first_mismatch = None;
    let mut forced_first_logit_div = None;
    let mut forced_first_argmax_div = None;
    let mut forced_max_logit_err = None;
    if do_greedy {
        let want: Vec<u32> = require_i64(&dump, "greedy_ids")
            .iter()
            .map(|&i| u32::try_from(i).expect("token id"))
            .collect();
        assert_eq!(want.len(), 128);
        let ref_logits = require_f32(&dump, "greedy_logits");
        let v = model.vocab();
        assert_eq!(ref_logits.len(), 128 * v);

        // Greedy in the dumps used generate(pad_token_id=eos=7). Re-prefill with
        // that pad mask so teacher-forcing and free greedy follow the same path.
        let mut traces_m = llm_engine::LayerTraces::default();
        let mut cache = model.prefill_masked(
            &ids,
            Some(&mut traces_m),
            reserve,
            Some(GENERATE_PAD_TOKEN_ID),
        );
        // Teacher-force the reference tokens and compare logits at every step.
        let mut current = traces_m.logits[traces_m.logits.len() - v..].to_vec();
        let mut max_forced = 0.0f32;
        for step in 0..128 {
            let exp_row = &ref_logits[step * v..(step + 1) * v];
            let err = max_abs_err(&current, exp_row);
            if err > max_forced {
                max_forced = err;
            }
            let pred = argmax(&current) as u32;
            if err > LOGIT_TOL && forced_first_logit_div.is_none() {
                forced_first_logit_div = Some((step, err));
            }
            if pred != want[step] && forced_first_argmax_div.is_none() {
                forced_first_argmax_div = Some((step, pred, want[step]));
            }
            current = model.decode_one(want[step], &mut cache);
        }
        forced_max_logit_err = Some(max_forced);

        // Free-run greedy on a fresh cache (the teacher-forced cache followed `want`).
        let mut traces2 = LayerTraces::default();
        let mut cache2 = model.prefill_masked(
            &ids,
            Some(&mut traces2),
            reserve,
            Some(GENERATE_PAD_TOKEN_ID),
        );
        let last = &traces2.logits[traces2.logits.len() - v..];
        let mut next = argmax(last) as u32;
        let mut got = Vec::with_capacity(128);
        for (step, &exp) in want.iter().enumerate() {
            if next != exp && greedy_first_mismatch.is_none() {
                greedy_first_mismatch = Some((step, next, exp));
            }
            got.push(next);
            let logits = model.decode_one(next, &mut cache2);
            next = argmax(&logits) as u32;
        }
        greedy_ok = Some(got == want);
    }

    PromptReport {
        name: name.to_string(),
        tokens: ids.len(),
        embed_err,
        layer_err,
        final_norm_err,
        logits_err,
        first_layer_over,
        layer_over_count,
        greedy_ok,
        greedy_first_mismatch,
        forced_first_logit_div,
        forced_first_argmax_div,
        forced_max_logit_err,
    }
}

fn print_report(r: &PromptReport) {
    println!(
        "=== {}  T={}  embed={:.3e}  final_norm={:.3e}  logits={:.3e} ===",
        r.name, r.tokens, r.embed_err, r.final_norm_err, r.logits_err
    );
    for (i, err) in r.layer_err.iter().enumerate() {
        let over = r.layer_over_count.get(i).copied().unwrap_or(0);
        let mark = if over > 0 { "  <-- OVER" } else { "" };
        println!("  layer_{i:02}: max_abs={err:.3e}  over={over}{mark}");
    }
    if let Some((i, err)) = r.first_layer_over {
        println!(
            "  first layer over atol={LAYER_ATOL} rtol={LAYER_RTOL}: layer_{i:02} (max abs {err:.3e})"
        );
    }
    if let Some(max_err) = r.forced_max_logit_err {
        match (r.forced_first_logit_div, r.forced_first_argmax_div) {
            (None, None) => println!("  teacher-forced 128: logits ok (max abs {max_err:.3e})"),
            (logit, argmax_div) => {
                print!("  teacher-forced 128: max abs {max_err:.3e}");
                if let Some((step, err)) = logit {
                    print!("  first logit div at step {step} ({err:.3e})");
                }
                if let Some((step, got, exp)) = argmax_div {
                    print!("  first argmax div at step {step}: got {got}, expected {exp}");
                }
                println!();
            }
        }
    }
    match r.greedy_ok {
        Some(true) => println!("  greedy 128: MATCH"),
        Some(false) => {
            if let Some((step, got, exp)) = r.greedy_first_mismatch {
                println!("  greedy 128: MISMATCH at step {step}: got {got}, expected {exp}");
            } else {
                println!("  greedy 128: MISMATCH");
            }
        }
        None => {}
    }
}

fn bench_decode(model: &Lfm2Model, ids: &[u32], steps: usize, threads: usize) -> f64 {
    with_threads(threads, || {
        let v = model.vocab();
        let mut traces = LayerTraces::default();
        let mut cache = model.prefill(ids, Some(&mut traces), ids.len() + steps);
        let tok = argmax(&traces.logits[traces.logits.len() - v..]) as u32;
        // Warmup one token so caches are touched.
        let _ = model.decode_one(tok, &mut cache);

        // Fresh prefill for the timed run.
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

/// Compared against Hugging Face reference dumps (eager, fp32).
///
/// Skips cleanly when the dumps or weights are absent (`LLMHOST_GOLDEN_DIR`,
/// `LLMHOST_MODEL_DIR`). Until decode exactly matches the reference greedy
/// path, this test is expected to fail when dumps are present: it is not
/// ignored.
#[test]
fn lfm2_700m_matches_golden() {
    let Some(dir) = golden_dir() else {
        eprintln!(
            "skipping lfm2_700m_matches_golden: no reference dumps found              (set LLMHOST_GOLDEN_DIR or place a manifest at tests/golden/lfm2-700m/)"
        );
        return;
    };
    if !dir.join("manifest.json").is_file() {
        eprintln!(
            "skipping lfm2_700m_matches_golden: {} has no manifest.json",
            dir.display()
        );
        return;
    }

    println!("loading LFM2-700M weights (LLMHOST_MODEL_DIR or the local Hugging Face Hub cache)…");
    let t0 = Instant::now();
    let model = match Lfm2Model::load_lfm2_700m() {
        Ok(model) => model,
        Err(err) => {
            eprintln!(
                "skipping lfm2_700m_matches_golden: LFM2-700M weights not found ({err});                  set LLMHOST_MODEL_DIR or use a local Hugging Face Hub cache"
            );
            return;
        }
    };
    println!(
        "loaded in {:.1}s  hidden={} vocab={} layers={} intermediate={}",
        t0.elapsed().as_secs_f64(),
        model.hidden(),
        model.vocab(),
        model.config.num_hidden_layers,
        model.config.intermediate_size
    );
    assert_eq!(model.config.intermediate_size, 6912);
    assert_eq!(model.config.head_dim, 64);

    let mut reports = Vec::new();
    let mut all_ok = true;
    for name in PROMPTS {
        if !dir.join(format!("{name}.npz")).is_file() {
            // Individual dumps may be absent even when the manifest is present.
            eprintln!("note: {name}.npz is not present; skipping that prompt");
            continue;
        }
        let do_greedy = GREEDY_PROMPTS.contains(name);
        let report = eval_prompt(&model, &dir, name, do_greedy);
        print_report(&report);
        if !report.layers_ok()
            || !report.logits_ok()
            || !report.force_feed_ok()
            || report.greedy_ok == Some(false)
        {
            all_ok = false;
        }
        if report.embed_err > 1e-6 {
            eprintln!(
                "warning: embed max-abs {} is larger than expected",
                report.embed_err
            );
        }
        reports.push(report);
    }

    let short = read_npz(dir.join("short_en.npz")).expect("short_en npz");
    let ids: Vec<u32> = require_i64(&short, "ids")
        .iter()
        .map(|&i| u32::try_from(i).unwrap())
        .collect();
    drop(short);
    let steps = 64;
    let tps1 = bench_decode(&model, &ids, steps, 1);
    let tps8 = bench_decode(&model, &ids, steps, 8);
    println!("decode tok/s (fp32, informational, {steps} tokens after prefill of short_en):");
    println!("  1-thread: {tps1:.2}");
    println!("  8-thread: {tps8:.2}");

    if !all_ok {
        let mut msg = String::from("M1 golden gate failed:\n");
        for r in &reports {
            if let Some((i, err)) = r.first_layer_over {
                msg.push_str(&format!(
                    "  {}: first diverging layer is layer_{i:02} (max abs {err:.3e}, atol+rtol gate)\n",
                    r.name
                ));
                for (j, e) in r.layer_err.iter().enumerate() {
                    msg.push_str(&format!("    layer_{j:02}: {e:.3e}\n"));
                }
            } else if !r.logits_ok() {
                msg.push_str(&format!(
                    "  {}: logits max abs {:.3e} > {LOGIT_TOL}\n",
                    r.name, r.logits_err
                ));
            }
            if let Some((step, err)) = r.forced_first_logit_div {
                msg.push_str(&format!(
                    "  {}: teacher-forced logits first diverge at step {step} (max abs {err:.3e} > {LOGIT_TOL})\n",
                    r.name
                ));
            }
            if let Some((step, got, exp)) = r.forced_first_argmax_div {
                msg.push_str(&format!(
                    "  {}: teacher-forced argmax first diverges at step {step}: got {got}, expected {exp}\n",
                    r.name
                ));
            }
            if r.greedy_ok == Some(false) {
                msg.push_str(&format!(
                    "  {}: greedy mismatch {:?}\n",
                    r.name, r.greedy_first_mismatch
                ));
            }
        }
        panic!("{msg}");
    }
}
