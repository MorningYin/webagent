---
name: message-array-and-token-budget
description: "Stage-4 build_master_with_segments.py reconstructs lossless master from images+responses, inlines segment meta-calls; plus Qwen3-VL token-budget numbers per window K"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

Stage 4 of the segment-SFT pipeline. Script: `/root/webgym/scripts/build_master_with_segments.py`.

## Why reconstruction is needed (not reuse the stored prompt)
The rollout's per-step stored `prompt` is a BOUNDED-WINDOW context (~5 recent screenshots + a running_log), NOT full history. Confirmed: the last-step prompt has ~5 images and 4 assistant turns regardless of num_steps (21/41/47...). So the lossless master MUST be RECONSTRUCTED from each step's `image_path` + `response`, never from the stored prompt window.

## Master message layout (per trajectory)
`[system]` then, for each step i: `[user: image step_i (+ task text only when i==0)]` `[assistant: response_i]`. So message count == `1 + 2*num_steps`. Images are referenced by RELATIVE path `images/<task>/step_k.png` via `st["image_path"]` (image_url), NOT base64 — so the `images/` dir travels alongside the jsonl. (`build_one`, msgs assembled lines ~148-156.)

## What build_master_with_segments.py adds on top
1. Injects `segment_beginner(observation, purpose)` + `segment_stopper(summary, result)` tool specs into the system `<tools>` block (`META_TOOLS` inserted before `</tools>`; `augment_system`).
2. Appends a detailed `# Working in segments` spec to the system prompt (`SEG_INSTRUCTION`): segment concept, the boundary rule (start a new segment only when gap/candidate/operation/target-state changes — NOT on engine/URL/scroll/retry change), the four-field meanings, the "no hindsight / descriptive not judgement" rule, and the per-step ordering.
3. INLINES the meta-calls into boundary-step assistant outputs (field values come from `segments.jsonl`):
   - segment START step: PREPEND `segment_beginner` FIRST, before Thought/Action/computer_use.
   - segment END step: APPEND `segment_stopper` LAST, after the action's tool_call.
   - A length-1 segment carries BOTH (beginner first, stopper last) around its single step. Middle steps carry NEITHER.
   - Meta-calls are bookkeeping-only (no browser action), written inline as `<tool_call>{"name":..., "arguments":...}</tool_call>` text in assistant content, matching the existing inline tool-call format (`tool_call()` helper).

## Gotchas
Some steps have `response == None` (key present, value null) → use `(st.get("response") or "")` to avoid TypeError. Both the resp dict build and the final assistant write apply this guard.

## Output
`messages_with_segments.jsonl`: 2706 records, each `{task_id, reward, num_steps, n_segments, source, messages}`. All msg-counts correct; beginner==stopper==segments==16386; 66285 image refs.

## Token budget (results recorded; estimate_tokens.py was a scratch utility, now deleted)
Downstream segment-aware per-step input = system(+segment doc, ~2323 tok fixed) + task + A(history of prior CLOSED segments, purpose->result) + current-segment beginner header + B(within-segment summary: join of rolling notes for steps scrolled out of the K-image window) + K screenshots&actions + response. Model = Qwen3-VL-8B. Step-level SFT = 66205 samples.

Image tokens: 1280x768 at Qwen3-VL patch16 * merge2 (32px/token) = 40x24 = **960 tokens/screenshot**.

Totals over the whole set (text measured with a Qwen2.5 tokenizer, same family = estimate):
- K=1 -> 266.3M total (text 202.7M + img 63.6M), avg 4022/sample
- K=2 -> 320.8M total (text 209.4M + img 111.4M), avg 4846/sample (text 3163 + img 1682)
- K=3 -> 363.0M total, avg 5483/sample

Images are the marginal driver: each +1 in K adds ~45M image tokens, while text stays ~flat (a larger window just swaps B-summary text for screenshots). If the VL actually uses factor-28 (patch14) -> 1242 tok/image, multiply the image term by ~1.29 (K=2 -> ~360M total).

**Why:** the bounded rollout window means the lossless master must be rebuilt from images+responses; and the K-window choice is dominated by image-token cost, so budgeting hinges on the 960-tok/screenshot figure and how many screenshots K keeps in context.
**How to apply:** to (re)generate the master, run build_master_with_segments.py over dataset.jsonl + segments.jsonl (keep images/ alongside the output); to size a training run, scale per the K-table above (text ~flat, +45M img per K) and re-derive image tokens from resolution / patch / merge factor if the model config differs.

Cross-link: [[segment-sft-pipeline]], [[clean-dataset-artifacts]], [[segmenter-design]], [[within-segment-summary-design]].
