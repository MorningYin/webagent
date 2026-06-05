#!/usr/bin/env python
"""Master message array WITH segment_beginner / segment_stopper meta-calls inlined.

Builds the loss-free master conversation (system + task + every screenshot +
every assistant response) and then:

  1. Adds segment_beginner / segment_stopper tool specs to the system <tools>.
  2. Inlines the meta-calls into the boundary steps' assistant output:
       - segment start step : prepend  segment_beginner(observation, purpose)
       - segment end   step : append   segment_stopper(summary, result)
     A length-1 segment gets BOTH around its single assistant turn.

The four field values come from segments.jsonl (observation/purpose/summary/
result). Meta-calls are bookkeeping only — they do not operate the browser and
do not add new turns.

Downstream (other machine) assembles the final per-step training input from
this message array + seg_histories_dual.jsonl (within-segment summary) + images.

Output: messages_with_segments.jsonl  {task_id, reward, num_steps, n_segments, messages:[...]}
"""
import os, json, argparse

META_TOOLS = '''{
  "name": "segment_beginner",
  "description": "Meta-call that OPENS a skill segment (a contiguous run of actions pursuing one objective). It does NOT operate the browser and does not consume a browser step. Emit it as the FIRST thing in your output at the moment you begin pursuing a new objective, before your Thought/Action. Fill it using only what is known BEFORE acting — no hindsight.",
  "parameters": {
    "type": "object",
    "properties": {
      "observation": {
        "type": "string",
        "description": "The state observed right now, before this segment acts: where you are and what is or is not yet known that is relevant to the task."
      },
      "purpose": {
        "type": "string",
        "description": "The state transition this segment intends to achieve (a sub-goal), described as an intended outcome — not the concrete UI operation."
      }
    },
    "required": ["observation", "purpose"]
  }
}
{
  "name": "segment_stopper",
  "description": "Meta-call that CLOSES the current skill segment. It does NOT operate the browser and does not consume a browser step. Emit it as the LAST thing in your output on the step where the current objective is achieved or confirmed blocked, after your action's tool_call.",
  "parameters": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "What was actually done within this segment — the process and actions taken."
      },
      "result": {
        "type": "string",
        "description": "The resulting state after this segment: what was obtained, or what blocked it."
      }
    },
    "required": ["summary", "result"]
  }
}'''

SEG_INSTRUCTION = """

# Working in segments

Your trajectory is organized into **skill segments**. A segment is a contiguous run of one or more steps that pursues a **single objective** — one intended change of state (one "skill"), such as reaching a particular page, locating a candidate, inspecting one item, or extracting one fact. Segments are head-to-tail: every step belongs to exactly one segment, and a new segment begins exactly where the previous one ends. Think and act one segment at a time.

## What counts as one segment

A segment is defined by the *state transition it attempts*, not by the website, path, or tactic you happen to use. Stay inside the same segment as long as you are still pursuing the same objective — even if you switch source, change route, retry, or recover from a block. The objective, not the mechanics, is the unit.

Start a NEW segment only when the local state-transition attempt itself changes — i.e. when any of these changes:
- the **gap** you are trying to close (the sub-goal), or
- the **candidate** you are working on (a different item/result/page than before), or
- the **operation** (e.g. moving from discovery → inspection, inspection → extraction, extraction → submission), or
- the **target state** you are trying to reach.
Do NOT start a new segment merely because you changed search engine, URL, scroll position, or because one attempt failed and you retried. Those stay in the current segment.

## Two bookkeeping meta-calls

Mark segment boundaries with two meta-calls. They are **bookkeeping only**: they do NOT operate the browser, do not move the mouse, and do not consume a browser step. They simply open and close a segment around your normal Thought / Action / computer_use block.

- **segment_beginner(observation, purpose)** — OPENS a segment. Emit it as the FIRST thing in your output on the step where you begin a new objective, before your Thought.
  - `observation`: the state you see right now, *before* this segment acts — where you are and what is or is not yet known that is relevant to the task.
  - `purpose`: the state transition this segment intends to achieve, stated as an intended *outcome* (a sub-goal), not as a UI operation.
  - Fill both using ONLY what is knowable before acting. Never use hindsight — do not describe what you will later discover.

- **segment_stopper(summary, result)** — CLOSES the current segment. Emit it as the LAST thing in your output on the step where the current objective is achieved or confirmed blocked, after your action's tool_call.
  - `summary`: what was actually done within this segment — the process and the actions taken.
  - `result`: the resulting state after this segment — what was obtained, or what blocked it.

All four fields are **descriptive records, not judgements**: do not rate quality, usefulness, correctness, or reward. Just describe the before-state, the intent, the process, and the after-state plainly.

## Ordering on a step

On a segment's first step, output in this order: `segment_beginner` → Thought → Action → `computer_use` (your real action). On a segment's last step: Thought → Action → `computer_use` → `segment_stopper`. A one-step segment carries BOTH — `segment_beginner` first and `segment_stopper` last, wrapped around the single Thought/Action/computer_use. Steps in the middle of a segment carry NEITHER meta-call: just Thought / Action / computer_use as usual.

This mirrors the dependencies you should respect: beginner is a function of (memory so far + current screen); stopper is a function of (this segment's beginner + the actions you actually took)."""


def task_text(prompt):
    for m in json.loads(prompt):
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "text":
                    return p["text"]
    return ""


def system_text(prompt):
    m = json.loads(prompt)
    return m[0]["content"] if m and m[0].get("role") == "system" else ""


def augment_system(sysc):
    """Insert the two meta-tool specs before </tools> and append the instruction."""
    marker = "</tools>"
    if marker in sysc:
        sysc = sysc.replace(marker, META_TOOLS + "\n" + marker, 1)
    return sysc + SEG_INSTRUCTION


def tool_call(name, args):
    return '<tool_call>\n' + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + '\n</tool_call>'


def build_one(rec, segs):
    steps = rec.get("steps", [])
    if not steps:
        return None, 0
    sysc = augment_system(system_text(steps[0]["prompt"]))
    task_c = task_text(steps[0]["prompt"])

    # step number -> per-step assistant response (start from the raw response)
    resp = {st.get("step"): (st.get("response") or "") for st in steps}

    # apply segment boundaries to the boundary steps' responses
    for sg in segs:
        a, b = sg["start_step"], sg["end_step"]
        if a in resp:
            beg = tool_call("segment_beginner",
                            {"observation": sg.get("observation", ""), "purpose": sg.get("purpose", "")})
            resp[a] = beg + "\n\n" + resp[a]
        if b in resp:
            stp = tool_call("segment_stopper",
                            {"summary": sg.get("summary", ""), "result": sg.get("result", "")})
            resp[b] = resp[b] + "\n\n" + stp

    messages = [{"role": "system", "content": sysc}]
    for i, st in enumerate(steps):
        img = {"type": "image_url", "image_url": {"url": st.get("image_path")}}
        if i == 0:
            user_content = [img, {"type": "text", "text": task_c}]
        else:
            user_content = [img]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": resp.get(st.get("step")) or ""})
    return messages, len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/dataset.jsonl")
    ap.add_argument("--segments", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/segments.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/webgym_runs/export/clean_dataset/messages_with_segments.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated task_ids for spot testing")
    args = ap.parse_args()

    seg_by_task = {}
    with open(args.segments) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or not r.get("segments"):
                continue
            seg_by_task[r.get("task_id")] = r["segments"]

    only = set(x for x in args.only.split(",") if x)
    n = skipped = noseg = 0
    with open(args.dataset) as f, open(args.out, "w") as fo:
        for line in f:
            r = json.loads(line)
            tid = r.get("task_id")
            if only and tid not in only:
                continue
            segs = seg_by_task.get(tid)
            if not segs:
                noseg += 1
                continue
            msgs, ns = build_one(r, segs)
            if msgs is None:
                skipped += 1
                continue
            fo.write(json.dumps({
                "task_id": tid, "reward": r.get("reward"),
                "num_steps": r.get("num_steps"), "n_segments": ns,
                "source": r.get("source"), "messages": msgs,
            }, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} ({skipped} empty, {noseg} no-segments) -> {args.out}")


if __name__ == "__main__":
    main()
