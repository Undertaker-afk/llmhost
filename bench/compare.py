#!/usr/bin/env python3
"""Run llmhost-bench and llama-bench with the same settings and print one table.

  python3 bench/compare.py --gguf path/to/LFM2-700M-Q4_0.gguf \
      --llama-bench path/to/llama-bench --threads 4,8 --prompt 512 --gen 128 --runs 5

Raw rows from both engines are appended to --out (JSONL). A row during which other
processes kept more than --max-load CPUs busy is marked noisy and is not counted as a result.
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", action="append", default=[], help="GGUF for llama.cpp (repeatable)")
ap.add_argument("--llama-bench", default=os.environ.get("LLAMA_BENCH", "llama-bench"))
ap.add_argument("--threads", default="8")
ap.add_argument("--prompt", type=int, default=512)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--max-load", type=float, default=0.5, help="max CPUs busy with other processes")
ap.add_argument("--out", default="bench/results.jsonl")
ap.add_argument("--skip-llmhost", action="store_true")
args = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []

def load():
    return float(open("/proc/loadavg").read().split()[0])

def busy():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v) - v[3] - v[4], sum(v)

def children_jiffies():
    t = os.times()
    return (t.children_user + t.children_system) * os.sysconf("SC_CLK_TCK")

if not args.skip_llmhost:
    tmp = os.path.join(root, "bench", ".llmhost-rows.jsonl")
    if os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run(["cargo", "run", "-q", "--release", "-p", "llmhost-bench", "--",
                    "--threads", args.threads, "--prompt", str(args.prompt), "--gen", str(args.gen),
                    "--runs", str(args.runs), "--max-load", str(args.max_load), "--out", tmp],
                   cwd=root, check=True)
    rows += [json.loads(l) for l in open(tmp)]
    os.remove(tmp)

for g in args.gguf:
    lb = load(); b0, t0 = busy(); c0 = children_jiffies()
    p = subprocess.run([args.llama_bench, "-m", g, "-t", args.threads, "-p", str(args.prompt),
                        "-n", str(args.gen), "-r", str(args.runs), "-o", "jsonl"],
                       capture_output=True, text=True, check=True)
    la = load(); b1, t1 = busy(); c1 = children_jiffies()
    other = max(0.0, (b1 - b0) - (c1 - c0)) / max(1, t1 - t0) * os.cpu_count()
    for l in p.stdout.splitlines():
        r = json.loads(l)
        test = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
        rows.append({
            "engine": "llama.cpp", "model": os.path.basename(g), "weights": r.get("model_type", ""),
            "test": test, "threads": r["n_threads"], "runs": len(r.get("samples_ts", [])),
            "tok_s_mean": r["avg_ts"], "tok_s_stdev": r["stddev_ts"], "samples": r.get("samples_ts", []),
            "load_before": lb, "load_after": la, "other_cpus": other, "noisy": other > args.max_load,
            "commit": r.get("build_commit", ""), "cpu": r.get("cpu_info", ""), "unix_time": int(time.time()),
        })

with open(os.path.join(root, args.out), "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"| engine | model / weights | test | threads | tok/s | other CPUs busy | |")
print(f"|---|---|---|---:|---:|---:|---|")
for r in sorted(rows, key=lambda r: (r["test"][:2], r["threads"], r["engine"])):
    flag = "noisy" if r["noisy"] else ""
    print(f"| {r['engine']} | {r['model']} {r['weights']} | {r['test']} | {r['threads']} | "
          f"{r['tok_s_mean']:.1f} ± {r['tok_s_stdev']:.1f} | {r['other_cpus']:.2f} | {flag} |")
