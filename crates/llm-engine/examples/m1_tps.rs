//! Informational LFM2-700M fp32 throughput: decode on `short_en`, prefill on `long_en`.

use std::path::PathBuf;
use std::time::Instant;

use llm_cpu::{argmax, with_threads};
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

fn decode_tps(model: &Lfm2Model, ids: &[u32], steps: usize, threads: usize) -> f64 {
    with_threads(threads, || {
        let v = model.vocab();
        // Warmup: one prefill and one decode step so caches are touched.
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
        // Warmup
        let _ = model.prefill(ids, None, ids.len());
        let t0 = Instant::now();
        let _ = model.prefill(ids, None, ids.len());
        let dt = t0.elapsed().as_secs_f64();
        (ids.len() as f64 / dt, dt)
    })
}

fn main() {
    println!("loading LFM2-700M fp32…");
    let t0 = Instant::now();
    let model =
        Lfm2Model::load_lfm2_700m().expect("LFM2-700M weights (LLMHOST_MODEL_DIR or Hub cache)");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let short = load_ids("short_en");
    let long = load_ids("long_en");
    let decode_steps = 128;

    println!(
        "decode: short_en prefill T={} then {decode_steps} incremental tokens (fp32)",
        short.len()
    );
    let dec1 = decode_tps(&model, &short, decode_steps, 1);
    println!("  1-thread decode: {dec1:.2} tok/s");
    let dec8 = decode_tps(&model, &short, decode_steps, 8);
    println!("  8-thread decode: {dec8:.2} tok/s");

    println!("prefill: long_en T={} (fp32)", long.len());
    let (pre1, dt1) = prefill_tps(&model, &long, 1);
    println!("  1-thread prefill: {pre1:.2} tok/s  ({dt1:.3}s)");
    let (pre8, dt8) = prefill_tps(&model, &long, 8);
    println!("  8-thread prefill: {pre8:.2} tok/s  ({dt8:.3}s)");
}
