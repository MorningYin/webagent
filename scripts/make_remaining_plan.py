#!/usr/bin/env python3
"""
Build a 'remaining tasks' plan from a partial run: keep task_ids that already
produced a REAL trajectory (>= min-steps), re-run everything else (quota/error
failures + never-run tasks). Writes a jsonl you can point train_tasks at.

Usage:
  python make_remaining_plan.py \
    --save-path /root/autodl-tmp/webgym_runs/run5k \
    --tasks-jsonl <data_path>/train.jsonl \
    --out /root/autodl-tmp/webgym_runs/remaining.jsonl \
    [--min-steps 2]
"""
import argparse, glob, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--tasks-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-steps", type=int, default=2,
                    help="trajectories with >= this many steps count as 'done' (1-step = quota/error garbage)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import torch
    done = set()           # task_ids with a real (multi-step) trajectory
    seen_total = real = 0
    for f in sorted(glob.glob(os.path.join(args.save_path, f"{args.split}_trajectories", "*.pt*"))):
        d = torch.load(f, weights_only=False)
        trajs = d["trajectories"] if isinstance(d, dict) else d
        for t in trajs:
            seen_total += 1
            try:
                tid = str(t[0]["observation"].task.task_id)
            except Exception:
                continue
            if len(t) >= args.min_steps:
                done.add(tid); real += 1

    # Filter the original task file to rows NOT already done.
    kept = 0; total = 0
    with open(args.tasks_jsonl) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if str(row.get("task_id")) in done:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"saved trajectories scanned : {seen_total}")
    print(f"real (>= {args.min_steps} steps) -> done    : {real}  (unique task_ids {len(done)})")
    print(f"original tasks             : {total}")
    print(f"remaining to (re)run       : {kept}  -> {args.out}")

if __name__ == "__main__":
    main()
