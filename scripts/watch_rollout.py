#!/usr/bin/env python3
"""
Live rollout monitor — progress + content, terminal-only (no web port needed).

Usage:
  python watch_rollout.py [--save-path DIR] [--port 6006] [--interval 5] [--content N]

- Progress comes live from the TaskMonitor JSON API (http://localhost:<port>/api/status).
- Content comes from incrementally-saved trajectories in <save-path>/{train,test}_trajectories/.
  (rollout configs here set save_traj_progress: true, so finished work shows up while running.)

--content N : also print a one-line content summary of the last N finished trajectories
              (task name, #steps, final action/answer parsed from the last gpt-5.4 response).
"""
import argparse, glob, json, os, time, urllib.request

def get_progress(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/status", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"_error": str(e)}

def load_latest_trajectories(save_path):
    trajs = []
    for split in ("train", "test"):
        files = sorted(glob.glob(os.path.join(save_path, f"{split}_trajectories", "*.pt*")))
        for f in files:
            try:
                import torch
                d = torch.load(f, weights_only=False)
                trajs += (d["trajectories"] if isinstance(d, dict) else d)
            except Exception:
                pass
    return trajs

def summarize_traj(t):
    try:
        obs0 = t[0].get("observation")
        task = getattr(getattr(obs0, "task", None), "task_name", "?")
    except Exception:
        task = "?"
    last = t[-1]
    resp = last.get("response")
    at = getattr(resp, "answering_tokens", {}) or {}
    action = (at.get("action") or "").strip().replace("\n", " ")[:80]
    return f"[{len(t)} steps] {str(task)[:50]:50s} | final action: {action}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default="/root/autodl-tmp/webgym_runs/run5k")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--content", type=int, default=0, help="show last N finished trajectories' content")
    args = ap.parse_args()

    while True:
        p = get_progress(args.port)
        ts = time.strftime("%H:%M:%S")
        if "_error" in p:
            print(f"[{ts}] dashboard not reachable on :{args.port} ({p['_error']}) — rollout may still be starting")
        else:
            s = p.get("summary", {})
            total = p.get("total_tasks", "?")
            done = s.get("finished", 0); failed = s.get("failed", 0); running = s.get("running", 0)
            waiting = s.get("waiting", 0) + s.get("not_started", 0)
            hms = s.get("elapsed_hms", "")
            print(f"[{ts}] {hms} | done {done} / {total}  (failed {failed}, running {running}, waiting {waiting})"
                  f"  | acting {s.get('action',0)} shooting {s.get('screenshot',0)} reward {s.get('reward',0)}")

        if args.content:
            trajs = load_latest_trajectories(args.save_path)
            if trajs:
                print(f"   ---- last {min(args.content,len(trajs))} of {len(trajs)} saved trajectories ----")
                for t in trajs[-args.content:]:
                    print("   " + summarize_traj(t))
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
