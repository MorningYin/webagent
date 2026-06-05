---
name: within-segment-summary-design
description: "Stage 3 build_seg_summaries.py — dual per-step DELTA notes (structured + mini) concatenated by prefix to form within-segment memory block \"B\""
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

`scripts/build_seg_summaries.py` (Stage 3) → output `seg_histories_dual.jsonl`. Builds the within-segment memory (block "B") used by the segment SFT pipeline.

Core idea — DESIGN B (per-step DELTA + concatenation):
- The within-segment memory for a target step is built LATER by PREFIX CONCATENATION of per-step DELTA notes: `history = join(rolling_*[0 : j-k])`, where `j` is the within-segment position and `k` is a randomized sliding-window size `k ∈ {1,2,3,4,5}`, `k_min=1`. The stopper step counts as a within-segment step (deepest lookup).
- DESIGN B (per-step delta notes, concatenated downstream) was chosen over cumulative self-contained notes because gpt-5.4-mini is too weak to write cumulative self-contained notes. Each note describes only its own event; the SYSTEM prompt explicitly tells mini "Each note describes only that event, not the cumulative prefix. Later code will concatenate earlier notes."

Per-segment output (gated on `L >= 2`, `MIN_L=2`; `L<2` segments get empty lists / no notes). TWO ALIGNED versions are stored deliberately to construct diversified data, each with `len == segment length`:
- `rolling_structured[m]` = deterministic event line: `[i] from=<title@url> -> action=<rendered> -> to=<title@url> ; surface=<url_changed|title_changed|same|final|unknown> ; thought=<~220 chars>`. `surface()` compares step m vs m+1 (page/source change ONLY, not task progress); `final` when the action is `answer`.
- `rolling_model[m]` = gpt-5.4-mini compressed natural-language one-sentence delta note for that same event.

Model / endpoint:
- Model = `gpt-5.4-mini`, yunwu ONLY (`YUNWU_BASE=https://yunwu.ai/v1`; security constraint — mini only exists on yunwu). Streaming SSE (`call_mini`). `--via-proxy` routes through the local LiteLLM proxy whose gpt-5.4-mini deployment is itself yunwu-only, so via-proxy stays naturally yunwu-only. See [[webgym-closed-api-rollout-setup]].

KEY DESIGN DECISION / bug history (surface=same facts):
- An earlier version did regex fact/issue extraction from `thought`; this was a REGRESSION — a buggy `_QUOTED` regex mis-paired English apostrophes in "I'm"/"Apple's" as quote delimiters, producing garbage facts like "m on Apple" and DROPPING real values like "M5" / "18 hours battery".
- Fix: feed mini the FULL thought (not regex-extracted fields) and instruct it via priority rules to capture concrete facts/values/specs/names/numbers/prices/dates from `thought` EVEN WHEN `surface=same`; only emit "no new task-relevant information was obtained" when there is no concrete fact AND no useful page/source change AND no obstacle. (Confirmed in current SYSTEM prompt: priority rule 1 overrides surface=same.) Verified on task 178188 seg1: now captures "Now supercharged by M5" and "Up to 18 hours of battery life" on surface=same scroll steps.

LONG-SEGMENT FIX:
- mini cannot reliably return N notes for very long segments (L=43..197) in one JSON object — it truncates/miscounts (`build_one` only retries twice, then leaves `rolling_model` empty). The full run left 12 such segments empty; backfilled by CHUNKING events (~24 per chunk) and concatenating, guaranteeing `len(notes)==len(events)`. Final state: 0 empty, 0 misaligned, 11297 segments with L>=2, ~61k mini notes.

**Why:** Block "B" must be reconstructable at any cut point via prefix concat under a randomized k-window, and mini is too weak for cumulative notes — so delta notes + downstream concatenation is the only reliable construction; capturing facts on surface=same is essential or the memory drops the actual answer values.
**How to apply:** Edit `scripts/build_seg_summaries.py` for note generation; keep dual output aligned (`len==L`), `MIN_L=2`, mini-on-yunwu, and the priority-rule-1 "facts even when surface=same" SYSTEM instruction; for L>=~40 segments use chunked generation to avoid truncation; consumed downstream by the SFT builder via `rolling_*[0:j-k]`. Cross-link: [[segment-sft-pipeline]], [[clean-dataset-artifacts]], [[segmenter-design]], [[message-array-and-token-budget]], [[webgym-closed-api-rollout-setup]].
