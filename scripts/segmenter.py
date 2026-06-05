#!/usr/bin/env python
"""Segmenter: flatten a web-agent trajectory and ask a strong model (gpt-5.5 on dkyx)
to cut it into contiguous, head-to-tail SEGMENTS — each one a state transition.

This is PURELY DESCRIPTIVE (not evaluative). For each segment:
  observation = state BEFORE the transition (situation at the segment's first step)
  purpose     = the INTENDED transition (forward-looking intent of the segment)
  result      = the RESULT of the transition (state the segment ends in)
  summary     = the PROCESS of the transition (flat narrative of the steps)

Input : clean_dataset/dataset.jsonl  (per-trajectory, each step has thought/action/url/page_title)
Output: segments.jsonl               (one line per trajectory: {task_id, segments:[...]})

Calls dkyx (newapi) directly — the only gateway carrying gpt-5.5 — with key rotation,
verify=False (relay chain), and bounded retry. Offline batch: does NOT touch the rollout proxy.

Usage:
  source scripts/start_litellm.sh   # exports LITELLM_DKYX_BASE + POLICY_API_KEY_NEWAPI[2,3,4]
  python scripts/segmenter.py --in <dataset.jsonl> --out <segments.jsonl> [--limit N] [--model gpt-5.5] [--concurrency 8]
"""
import os, sys, json, argparse, threading, itertools, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

DKYX_BASE = os.environ.get("LITELLM_DKYX_BASE", "https://newapi.dkyx.cc/v1")
KEYS = [os.environ[k] for k in ("POLICY_API_KEY_NEWAPI","POLICY_API_KEY_NEWAPI2",
                                 "POLICY_API_KEY_NEWAPI3","POLICY_API_KEY_NEWAPI4") if os.environ.get(k)]
_key_cycle = itertools.cycle(KEYS) if KEYS else itertools.cycle([os.environ.get("POLICY_API_KEY_NEWAPI","")])
_key_lock = threading.Lock()
def next_key():
    with _key_lock: return next(_key_cycle)

SYSTEM = """
You relabel a web-agent trajectory into contiguous CREDIT-BEARING SKILL SEGMENTS.

A segment is the largest contiguous block of tool calls that still represents ONE locally judgeable task-state transition attempt.

It is not a primitive action and not the whole task.
It is one bounded attempt to move from the current task state to a more useful task state.

The purpose of segmentation is credit assignment:
a later external judge should be able to compare the segment's before-state, actions, and after-state and ask:
"Did this tool-call chain move the task into a more useful state, stall, repeat, or harm progress?"

You are not the judge.
Do not assign reward, correctness, usefulness, or success criteria.
Write only descriptive metadata that an agent could emit as:

BeginSegment(observation, purpose)
StopSegment(summary, result)

Core boundary rule:
Cut when the local state-transition attempt changes.

A local state-transition attempt is defined by:
- the unresolved gap being worked on;
- the candidate, source object, or obstacle being handled;
- the operation type: discover, reach, extract, verify, recover, or submit;
- the more useful task state the agent is trying to reach.

Do not cut merely because the agent changes source, website, path, or tactic.
A source/path switch is only a boundary if it changes the gap, candidate, operation, or target state.

Keep the same segment when different actions or sources are just tactics for the same precise attempt.
For example: trying the same clue phrase across a few search interfaces, retrying a blocked page, using direct URL after a click fails, or trying URL variants for the same intended artifact.

Start a new segment when the agent changes from discovery to inspection, inspection to extraction, extraction to verification, one candidate to another, one requirement to another, obstacle handling to information gathering, or information gathering to final answer submission.

Avoid over-fine segmentation:
do not split clicks, scrolls, typing, waiting, small retries, or tactical source switches that still serve the same precise attempt.

Avoid over-coarse segmentation:
do not merge different candidates, different unresolved requirements, different operation types, or many unrelated routes under vague purposes like "continue searching" or "find the answer".

Fields:

observation:
  The task-relevant state before the segment starts.
  Include current page/source/search context, known facts, unresolved requirements, active candidate, obstacle, and relevant failed routes.
  Use only information knowable at the segment start.

purpose:
  The intended local state transition, written as a forward-looking attempt.
  It is not a UI action, not a success condition, and not a scoring rule.
  Avoid vague purposes like "continue searching", "try more sources", or "find the answer".

summary:
  What the agent actually did inside the segment: actions, queries, route choices, retries, tactic switches, and recoveries.
  Do not evaluate quality or correctness.
  Do not hide repeated or failed attempts.

result:
  The ending task state after the segment's final action has taken effect.
  Record where the browser ended, what was visible or recorded, what changed, what remained missing, or how the route stalled.
  If no new task-relevant information was obtained, say so explicitly.
  Do not claim final success unless the trajectory submits a final answer.

Segments must be contiguous and cover every step exactly once.
Use inclusive start_step and end_step.

Output ONLY:
{"segments":[{"segment_id":0,"start_step":<int>,"end_step":<int>,"observation":"...","purpose":"...","summary":"...","result":"..."}]}
"""

def render_action(a):
    if not isinstance(a, dict): return "(none)"
    arg = a.get("arguments") or {}
    act = arg.get("action")
    c = arg.get("coordinate") or [None, None]
    if act == "left_click": return f"click({c[0]},{c[1]})"
    if act == "type":       return f'type({c[0]},{c[1]},{(arg.get("text") or "")[:80]!r})'
    if act == "scroll":     return f'scroll({arg.get("direction")})'
    if act == "navigate":   return f'navigate({(arg.get("url") or "")[:120]!r})'
    if act == "answer":     return f'answer({(arg.get("text") or "")[:200]!r})'
    if act == "go_back":    return "back()"
    if act == "wait":       return f'wait({arg.get("time")})'
    return f"{act}()"

def flatten(traj):
    """traj = dataset.jsonl record. Returns (flat_text, first_step, last_step)."""
    lines = [f"TASK: {(traj.get('task_name') or '')[:700]}", ""]
    steps = [s for s in traj["steps"] if (s.get("action") or s.get("thought"))]  # drop blank step 0
    idxs = []
    for s in steps:
        n = s.get("step")
        idxs.append(n)
        th = (s.get("thought") or "").replace("\n", " ").strip()[:140]
        pt = (s.get("page_title") or "").strip()[:80]
        lines.append(f"[{n}] {render_action(s.get('action'))} | page: {pt} | thought: {th}")
    if not idxs:
        return None, None, None
    return "\n".join(lines), idxs[0], idxs[-1]

def call_dkyx(model, system, user, timeout=300):
    """Streamed chat completion. Streaming keeps bytes flowing so the relay's
    openresty proxy_read_timeout resets per-chunk — long generations (300+ step
    trajectories) no longer hit a whole-response 504. Falls back across keys/retries."""
    body = {"model": model,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "temperature": 0.2, "response_format": {"type":"json_object"}, "stream": True}
    last = None
    for attempt in range(5):
        try:
            buf = []
            with httpx.Client(verify=False, timeout=httpx.Timeout(timeout, connect=10.0)) as cli:
                with cli.stream("POST", f"{DKYX_BASE}/chat/completions",
                                headers={"Authorization": f"Bearer {next_key()}"}, json=body) as r:
                    if r.status_code != 200:
                        last = f"HTTP {r.status_code}: {r.read()[:200]}"
                        continue
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        try:
                            delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                            if delta: buf.append(delta)
                        except Exception:
                            continue
            if buf:
                return "".join(buf)
            last = last or "empty stream"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"dkyx call failed after retries: {last}")

def validate(segs, first, last):
    """contiguous, head-to-tail, full coverage. Returns (ok, msg)."""
    if not segs: return False, "no segments"
    segs = sorted(segs, key=lambda s: s["start_step"])
    if segs[0]["start_step"] != first: return False, f"first start {segs[0]['start_step']} != {first}"
    if segs[-1]["end_step"] != last:  return False, f"last end {segs[-1]['end_step']} != {last}"
    for i, s in enumerate(segs):
        if s["end_step"] < s["start_step"]: return False, f"seg{i} end<start"
        if i > 0 and s["start_step"] != segs[i-1]["end_step"] + 1:
            return False, f"gap/overlap between seg{i-1} and seg{i}"
    return True, "ok"

def segment_one(traj, model):
    flat, first, last = flatten(traj)
    if flat is None:
        return {"task_id": traj.get("task_id"), "error": "no real steps", "segments": []}
    user = flat + f"\n\nSegment steps [{first}..{last}]. Return the JSON object."
    for attempt in range(2):
        raw = call_dkyx(model, SYSTEM, user)
        try:
            segs = json.loads(raw).get("segments", [])
        except Exception:
            user += "\n\n(Your last reply was not valid JSON. Return ONLY the JSON object.)"
            continue
        ok, msg = validate(segs, first, last)
        if ok:
            for i, s in enumerate(segs): s["segment_id"] = i
            return {"task_id": traj.get("task_id"), "source": traj.get("source"),
                    "reward": traj.get("reward"), "num_steps": traj.get("num_steps"),
                    "first_step": first, "last_step": last, "n_segments": len(segs),
                    "segments": segs}
        user += f"\n\n(Your segments were invalid: {msg}. They must be contiguous, head-to-tail, cover [{first}..{last}] exactly. Retry.)"
    return {"task_id": traj.get("task_id"), "error": f"invalid after retries: {msg}",
            "first_step": first, "last_step": last, "segments": segs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/dataset.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/segments.jsonl")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--via-proxy", action="store_true",
                    help="route through the local LiteLLM proxy (load-balances 4 dkyx keys) instead of calling dkyx directly")
    args = ap.parse_args()

    if args.via_proxy:
        global DKYX_BASE, _key_cycle
        DKYX_BASE = os.environ["POLICY_PROXY_BASE"].rstrip("/")          # http://localhost:4000/v1
        _key_cycle = itertools.cycle([os.environ["LITELLM_MASTER_KEY"]])  # proxy authenticates; it rotates dkyx keys
        print(f"via LiteLLM proxy: {DKYX_BASE}", flush=True)

    trajs = []
    with open(args.inp) as f:
        for line in f:
            trajs.append(json.loads(line))
            if args.limit and len(trajs) >= args.limit: break

    # resume: keep successfully-written task_ids, re-run errored/missing ones (append mode)
    done_ids = set()
    if os.path.exists(args.out):
        kept = []
        for line in open(args.out):
            try: r = json.loads(line)
            except Exception: continue
            if not r.get("error") and r.get("segments"):
                done_ids.add(r.get("task_id")); kept.append(line)
        with open(args.out, "w") as f: f.writelines(kept)
    trajs = [t for t in trajs if t.get("task_id") not in done_ids]
    print(f"loaded; already done {len(done_ids)}; to run {len(trajs)}; model={args.model}; -> {args.out}", flush=True)

    done = err = 0
    lock = threading.Lock()
    with open(args.out, "a") as fo, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(segment_one, t, args.model): t.get("task_id") for t in trajs}
        for fut in as_completed(futs):
            try: rec = fut.result()
            except Exception as e:
                rec = {"task_id": futs[fut], "error": f"{type(e).__name__}: {e}", "segments": []}
            with lock:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
                done += 1
                if rec.get("error"): err += 1
                if done % 10 == 0 or done == len(trajs):
                    print(f"  {done}/{len(trajs)}  (errors={err})", flush=True)
    print(f"DONE: {done} written, {err} errored -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
