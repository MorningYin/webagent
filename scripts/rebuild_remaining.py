#!/usr/bin/env python
"""Resume-safe task list rebuilder. Run BEFORE any rollout restart so we NEVER re-run
already-collected tasks.

Scans all banked trajectories under run_v2/train_trajectories/, collects the task_name of
every trajectory we already have REAL data for (steps >= 2 — success OR fail, both valuable),
and writes a new remaining list = (source pool) MINUS those. 1-step garbage (error trajectories)
is NOT counted as done -> it gets re-run.

Usage:
  python scripts/rebuild_remaining.py            # dry-run: just print counts
  python scripts/rebuild_remaining.py --write     # write remaining_resume.jsonl
"""
import torch, json, re, glob, os, sys

TRAJ_DIR = "/root/autodl-tmp/webgym_runs/run_v2/train_trajectories"
SOURCE_POOL = "/root/autodl-tmp/webgym_runs/remaining.jsonl"   # the pool this run draws from
OUT = "/root/autodl-tmp/webgym_runs/remaining_resume.jsonl"
MIN_STEPS = 2   # >=2 steps = a real attempt we keep; <2 = garbage, re-run

KEYLEN = 60   # task descriptions share their first chars; 60-char prefix is a robust, unique-enough key.
              # (raw_prompt "Task:" text == task_name + appended answer-format, so prefix-key aligns them;
              #  ~94% of completed tasks match — the rest are rewritten/decomposed tasks, re-run is harmless.)

def key(s):
    return re.sub(r'\s+', ' ', (s or '')).strip().lower()[:KEYLEN]

def task_of(traj):
    for s in traj:
        rp = getattr(s.get('response'), 'raw_prompt', '') or ''
        m = re.search(r'Task:\s*(.+)', rp)
        if m:
            return key(m.group(1))
    return None

done = set()
scanned = 0
for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.pt*"))):
    try:
        d = torch.load(f, weights_only=False)
    except Exception as e:
        print(f"  ! skip {os.path.basename(f)}: {e}"); continue
    ts = d['trajectories'] if isinstance(d, dict) else d
    n_real = 0
    for t in ts:
        if len(t) >= MIN_STEPS:
            tn = task_of(t)
            if tn:
                done.add(tn); n_real += 1
    scanned += len(ts)
    print(f"  {os.path.basename(f)}: {len(ts)} traj, {n_real} real(>= {MIN_STEPS} steps)")

print(f"\nscanned {scanned} trajectories | unique completed task-keys: {len(done)}")

pool = [json.loads(l) for l in open(SOURCE_POOL) if l.strip()]
remaining = [t for t in pool if key(t.get('task_name')) not in done]
skipped = len(pool) - len(remaining)
print(f"source pool {SOURCE_POOL}: {len(pool)} | already-done (skip) {skipped} | remaining to run {len(remaining)}")

if "--write" in sys.argv:
    with open(OUT, "w") as fo:
        for t in remaining:
            fo.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"WROTE {OUT} ({len(remaining)} tasks). Point config train_tasks -> remaining_resume.jsonl before restart.")
else:
    print("(dry-run; pass --write to emit remaining_resume.jsonl)")
