#!/usr/bin/env python
"""Build two aligned within-segment history lists with minimal preprocessing.

For each segment, this script outputs two DELTA history lists with the same length:

  rolling_structured[m] = deterministic event string for real step m
  rolling_model[m]      = mini-compressed natural-language note for real step m

Both are per-step deltas, not cumulative summaries.
Downstream builds old within-segment memory by prefix concatenation:

  history = join(rolling_*[0 : j-k])

Design principle:
  Do not do clever extraction here. Just make each event explicit:
    from-page -> action -> to-page ; surface-change ; thought
  Then mini only rewrites the event into a short note.

Input : dataset.jsonl   (steps have step/thought/action/url/page_title)
        segments.jsonl  (segments have start_step/end_step/purpose)
Output: seg_histories_dual.jsonl
"""
import os, json, argparse, threading, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit
import httpx

YUNWU_BASE = os.environ.get("LITELLM_YUNWU_BASE", "https://yunwu.ai/v1")
BASE = YUNWU_BASE
MODEL = "gpt-5.4-mini"
KEYS = [os.environ[k] for k in ("POLICY_API_KEY_YUNWU",) if os.environ.get(k)]
_key_cycle = itertools.cycle(KEYS) if KEYS else itertools.cycle([""])
_key_lock = threading.Lock()

MIN_L = 2

SYSTEM = """You rewrite compact per-step web-agent events into short natural-language DELTA notes.

Each event already states:
from page/source -> action -> to page/source ; surface change ; thought

Your job is simple:
write ONE short note for EACH event, in order.
Each note describes only that event, not the cumulative prefix.
Later code will concatenate earlier notes as old within-segment memory.

Important priority:
1. If the thought contains concrete task-relevant facts, names, numbers, prices, dates, specs, answer fragments, or candidates, include them in the note. This is true even if surface=same.
2. Otherwise, if the action reached a new page/source/route, record the new page/source/route.
3. Otherwise, if the action failed, repeated, was blocked, or produced no visible change, say so briefly.
4. Only say "no new task-relevant information was obtained" when there is no concrete fact and no useful page/source change.

Rules:
- Output exactly one note per event.
- Do not judge correctness, reward, usefulness, or quality.
- Do not invent facts.
- Do not mention future events.
- Keep each note one concise sentence.
- Do not include coordinates unless they explain a repeated failed UI click.

Output ONLY valid JSON:
{"rolling_model":["note 0","note 1"]}
The number of notes must equal the number of input events.
"""

def next_key():
    with _key_lock:
        return next(_key_cycle)


def clean(s, limit=None):
    """Simple whitespace cleanup only. No semantic extraction."""
    s = " ".join((s or "").replace("\n", " ").split()).strip()
    return s[:limit] if limit else s


def compact_url(url, limit=100):
    url = (url or "").strip()
    if not url:
        return ""
    try:
        sp = urlsplit(url)
        x = (sp.netloc + (sp.path or "") + (("?" + sp.query) if sp.query else "")).strip("/")
    except Exception:
        x = url
    return x if len(x) <= limit else x[:limit-1] + "…"


def source(step):
    if not isinstance(step, dict):
        return "unknown"
    title = clean(step.get("page_title"), 45) or "untitled"
    url = compact_url(step.get("url"), 75)
    return f"{title}@{url}" if url else title


def render_action(a):
    if not isinstance(a, dict):
        return "noop"
    arg = a.get("arguments") or {}
    act = arg.get("action")
    c = arg.get("coordinate") or [None, None]
    if act == "left_click":
        return f"click({c[0]},{c[1]})"
    if act == "type":
        return f"type({clean(arg.get('text'), 80)!r})"
    if act == "scroll":
        return f"scroll({arg.get('direction')})"
    if act == "navigate":
        return f"nav({compact_url(arg.get('url'), 100)!r})"
    if act == "answer":
        return f"answer({clean(arg.get('text'), 160)!r})"
    if act == "go_back":
        return "back()"
    if act == "wait":
        return f"wait({arg.get('time')})"
    return f"{act or 'action'}()"


def surface(before, after):
    """Surface is only page/source change, not task progress."""
    a = before.get("action") if isinstance(before, dict) else {}
    act = (a.get("arguments") or {}).get("action") if isinstance(a, dict) else None
    if act == "answer":
        return "final"
    if not after:
        return "unknown"
    bu, au = (before.get("url") or ""), (after.get("url") or "")
    bt, at = (before.get("page_title") or ""), (after.get("page_title") or "")
    if bu and au and bu != au:
        return "url_changed"
    if bt and at and bt != at:
        return "title_changed"
    return "same"


def event_line(local_i, step, after):
    """The structured history is this exact event line."""
    return (
        f"[{local_i}] from={source(step)} -> action={render_action(step.get('action'))} "
        f"-> to={source(after) if after else 'unknown_after'} ; "
        f"surface={surface(step, after)} ; thought={clean(step.get('thought'), 220) or 'none'}"
    )


def flatten_for_mini(purpose, events):
    lines = [f"PURPOSE: {clean(purpose, 260)}", "", "EVENTS:"]
    lines.extend(events)
    return "\n".join(lines)


def call_mini(system, user, timeout=180):
    body = {
        "model": MODEL,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": 0.2,
        "response_format": {"type":"json_object"},
        "stream": True,
    }
    last = None
    for _ in range(5):
        try:
            buf = []
            with httpx.Client(verify=False, timeout=httpx.Timeout(timeout, connect=10.0)) as cli:
                with cli.stream("POST", f"{BASE}/chat/completions",
                                headers={"Authorization": f"Bearer {next_key()}"}, json=body) as r:
                    if r.status_code != 200:
                        last = f"HTTP {r.status_code}: {r.read()[:200]}"
                        continue
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        d = line[5:].strip()
                        if d == "[DONE]":
                            break
                        try:
                            delta = json.loads(d)["choices"][0].get("delta", {}).get("content")
                            if delta:
                                buf.append(delta)
                        except Exception:
                            continue
            if buf:
                return "".join(buf)
            last = last or "empty stream"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"mini call failed: {last}")


def build_one(traj, segs):
    by_step = {s.get("step"): s for s in traj["steps"]}
    out = []
    for seg in segs:
        a, b = seg["start_step"], seg["end_step"]
        seg_steps = [by_step[n] for n in range(a, b + 1) if n in by_step]
        L = len(seg_steps)
        rec = {
            "segment_id": seg["segment_id"],
            "start_step": a,
            "end_step": b,
            "L": L,
            "rolling_structured": [],
            "rolling_model": [],
        }
        if L < MIN_L:
            out.append(rec)
            continue

        events = []
        for i, step in enumerate(seg_steps):
            global_step = step.get("step")
            # Use the real next global step as the after-state when available.
            # This keeps the final action of a segment grounded in the next observed page too.
            after = by_step.get(global_step + 1) if isinstance(global_step, int) else None
            events.append(event_line(i, step, after))

        rec["rolling_structured"] = events
        user = flatten_for_mini(seg.get("purpose", ""), events) + \
            f"\n\nReturn exactly {L} per-step DELTA notes as rolling_model, one for each event, indices 0..{L-1}."

        rolling = []
        for attempt in range(2):
            try:
                obj = json.loads(call_mini(SYSTEM, user))
                rolling = obj.get("rolling_model", [])
            except Exception as e:
                rec["error"] = str(e)
                rolling = []
            if isinstance(rolling, list) and len(rolling) == L:
                rec["rolling_model"] = [clean(str(x)) for x in rolling]
                rec.pop("error", None)
                break
            user += (
                f"\n\nYou returned {len(rolling) if isinstance(rolling, list) else 'invalid'} notes. "
                f"Return EXACTLY {L} per-step DELTA notes in rolling_model. Do not merge events."
            )
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/dataset.jsonl")
    ap.add_argument("--segments", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/segments.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/seg_histories_dual.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--via-proxy", action="store_true")
    args = ap.parse_args()

    global BASE, _key_cycle
    if args.via_proxy:
        BASE = os.environ["POLICY_PROXY_BASE"].rstrip("/")
        _key_cycle = itertools.cycle([os.environ["LITELLM_MASTER_KEY"]])
        print(f"via LiteLLM proxy: {BASE}", flush=True)

    ds = {}
    with open(args.dataset) as f:
        for line in f:
            d = json.loads(line)
            ds[d.get("task_id")] = d

    work = []
    with open(args.segments) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or not r.get("segments"):
                continue
            t = ds.get(r.get("task_id"))
            if t is None:
                continue
            work.append((r.get("task_id"), t, r["segments"]))
            if args.limit and len(work) >= args.limit:
                break

    done_ids = set()
    if os.path.exists(args.out):
        kept = []
        for line in open(args.out):
            try:
                rr = json.loads(line)
            except Exception:
                continue
            if rr.get("segments"):
                done_ids.add(rr.get("task_id"))
                kept.append(line)
        with open(args.out, "w") as f:
            f.writelines(kept)

    work = [w for w in work if w[0] not in done_ids]
    print(f"to build {len(work)} trajectories (already {len(done_ids)}); model={MODEL}; keys={len(KEYS)}; -> {args.out}", flush=True)

    done = err = 0
    lock = threading.Lock()
    with open(args.out, "a") as fo, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(build_one, t, segs): tid for tid, t, segs in work}
        for fut in as_completed(futs):
            try:
                segrecs = fut.result()
                rec = {"task_id": futs[fut], "segments": segrecs}
            except Exception as e:
                rec = {"task_id": futs[fut], "error": f"{type(e).__name__}: {e}", "segments": []}
            with lock:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                done += 1
                if any(s.get("error") for s in rec.get("segments", [])):
                    err += 1
                if done % 10 == 0 or done == len(work):
                    print(f"  {done}/{len(work)}  (traj with seg-errors={err})", flush=True)
    print(f"DONE: {done} written -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
