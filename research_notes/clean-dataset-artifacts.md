---
name: clean-dataset-artifacts
description: "Catalog of the WebGym clean_dataset export artifacts (jsonl files, images, counts, alignment keys) on /root/autodl-tmp"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

# clean_dataset export catalog

All artifacts live in `/root/autodl-tmp/webgym_runs/export/clean_dataset/` (NOT in git; ~17G total on disk). This is the consumable output of the segment-SFT pipeline (see [[segment-sft-pipeline]], [[data-migration-and-git-state]]). Image relative paths inside the jsonl files are rooted at this directory.

## Files (verified 2026-06-05)

- **dataset.jsonl** (765 MB, 2706 lines) — per-trajectory record.
  - Top keys: `task_id, source, harness, reward, num_steps, benchmark, domain, difficulty, task_name, steps`.
  - Each entry in `steps` has: `step, image_path` (relative `images/<task>/step_k.png`), `url, page_title, thought, action` (computer_use tool-call dict), `running_log, prompt, response`.
  - `prompt` is the rollout's BOUNDED-WINDOW message array serialized as a JSON string — only ~5 recent screenshots, NOT full history (see [[message-array-and-token-budget]]).

- **segments.jsonl** (13 MB, 2706 lines) — the four-field segmenter output (see [[segmenter-design]]).
  - Top keys: `task_id, source, reward, num_steps, first_step, last_step, n_segments, segments`.
  - Each segment: `segment_id, start_step, end_step, observation, purpose, summary, result`.

- **seg_histories_dual.jsonl** (32 MB, 2706 lines) — within-segment per-step delta notes (see [[within-segment-summary-design]]).
  - Top keys: `task_id, segments`.
  - Each segment: `segment_id, start_step, end_step, L, rolling_structured[...], rolling_model[...]`.
  - Only segments with `L>=2` carry notes (MIN_L=2). Two aligned versions: `rolling_structured` (deterministic) + `rolling_model` (gpt-5.4-mini).

- **messages_with_segments.jsonl** (100 MB, 2706 lines) — CANONICAL final master message array, with `segment_beginner`/`segment_stopper` tool calls inlined into assistant turns + segment doc in the system message.
  - Top keys: `task_id, reward, num_steps, n_segments, source, messages`.
  - `messages` roles: 1 `system`, then alternating `user`/`assistant` (one user + one assistant per step).

- **images/** (16G, 66285 PNG files, 1280x768 each).

- **MANIFEST.txt** and **TRANSFER_GUIDE.md** in the same dir.

## Key counts (verified, exact)

- Trajectories: **2706** (consistent line count across all four jsonl files).
- Steps: **66285** total (= PNG file count). avg **24.5** steps/traj.
- Segments total: **16386**, of which **11297** have `L>=2` (need within-segment notes) and **5089** are `L<2`.
- In messages_with_segments: exactly **16386** `segment_beginner` tool calls and **16386** `segment_stopper` tool calls in assistant turns. So beginner == stopper == total segments == sum of per-traj `n_segments` == **16386**.
  - (Substring grep overcounts: the two tool names also appear ~4x each in every system prompt as tool definitions; the load-bearing count is the `"name": "segment_X"` tool-call occurrences in assistant messages, which is 16386 each.)

## Alignment

- Primary join key is **`task_id`** everywhere.
- `segments.jsonl` and `seg_histories_dual.jsonl` share `segment_id` + `start_step`/`end_step`.
- In messages, the i-th assistant turn == step i (0-indexed, contiguous).
- `rolling_structured[p]` / `rolling_model[p]` == the p-th step inside that segment.
- Image relative paths are rooted at the clean_dataset directory.

## Notes

- `messages_master.jsonl` (~70 MB, plain master WITHOUT segments) still physically exists in the dir but is a superseded sub-step; its builder script `build_master_messages.py` was DELETED. The canonical artifact is **messages_with_segments.jsonl** — use that, not messages_master.
- The dir also contains earlier/pilot byproducts not part of the clean set: `messages_master.jsonl`, `sft_messages.jsonl` (~632 MB), `steps_io.jsonl` (~75 MB), `segments_oldprompt_0041.jsonl`, `segments_pilot.jsonl`, `segments_samples.jsonl`, `seg_histories_test2.jsonl`, `seg_summaries_test.jsonl`, plus logs and a `.ipynb_checkpoints/`. Ignore these for the canonical four-artifact + images set.
