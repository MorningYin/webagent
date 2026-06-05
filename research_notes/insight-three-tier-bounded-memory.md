---
name: insight-three-tier-bounded-memory
description: "Per-step context = three bounded memory tiers (history summary A / within-segment summary B / image sliding window); image tokens scale with segment length, not trajectory length; the tiers and beginner/stopper meta-calls are ONE segment-structured system"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

The central architectural idea: a policy step's context is assembled from THREE bounded memory tiers, each with its own independent boundary. This is the real point of segmentation — beyond credit assignment, it makes context cost scale with segment length instead of trajectory length.

## The three tiers (each bounded differently)

**A = History summary** — everything BEFORE the current segment's beginner, i.e. all CLOSED segments, compressed.
- Bound: roll over by SEGMENT COUNT (keep last N closed segments). Bound is on number of segments, not steps/tokens.
- Content per entry: purpose → result (goal / got) of that closed segment. The segment's `summary` (the mechanics / process of how it was done) is deliberately DROPPED from history, because history needs OUTCOMES, not process. Keeping the summary would roughly DOUBLE per-segment history tokens.
- Identity: each A entry literally IS a past segment's (purpose, result) pair = the output of a past `segment_stopper`.

**B = Within-segment summary** — the part AFTER the beginner: steps of the CURRENT segment that have already scrolled out of the image window, rendered as text.
- Bound: SEGMENT LENGTH. Because segments are short, B never grows with trajectory length.

**Sliding window** — the last k steps of the CURRENT segment as screenshots + actions (images).
- Bound: WITHIN-SEGMENT, k small.

## The payoff (token economics)

Because only the current segment's last-k screenshots are ever images, and everything older is text (A + B), **image tokens scale with SEGMENT LENGTH, not trajectory length**. A 300-step trajectory does not blow up image tokens. Text cost is ~flat across window size K; images are the marginal driver. Raising K just swaps B-text for screenshots. See [[message-array-and-token-budget]] for numbers (~960 img tokens/screenshot; K=2 ≈ 321M total).

## The unification (deep point)

The three-tier memory and the beginner/stopper meta-calls are NOT two systems — they are facets of ONE segment-structured memory:
- B is "what this segment has done so far," rolling. Finalize B at the segment's last step and it IS `stopper.summary`; the end-state is `stopper.result`. The model watches B grow and learns that `segment_stopper` = "finalize B."
- Symmetrically, every entry in A = a past `segment_stopper`'s product.
- The current segment's beginner header (observation, purpose) sits BETWEEN A and B.
- So A, B, the image window, and the beginner/stopper calls are all views of the same structure. See [[insight-policy-intrinsic-segmentation]].

## Implementation status

A/B/window assembly is the DOWNSTREAM (other-machine) step. The building blocks already exist: the segment four-tuple, dual within-segment rolling notes, and the lossless master with inlined meta-calls — see [[segment-sft-pipeline]]. Related: [[within-segment-summary-design]], [[segmenter-design]].

**Why:** Segmenting context into three bounded tiers decouples per-step token cost from trajectory length — image tokens track segment length only — while revealing that bounded memory and the beginner/stopper meta-calls are the same segment-structured system, so dropping process-summaries from history and rolling A by segment count is principled, not ad hoc.

**How to apply:** When assembling a policy step's context, build A from the last N closed segments as (purpose, result) only (drop their `summary`), build B as text from current-segment steps that scrolled past the k-image window, and keep only the last k current-segment screenshots as images; tune K to trade B-text for screenshots knowing text is ~flat and images dominate; treat `segment_stopper` as "finalize B → (summary, result)" and each A entry as a past stopper's output.
