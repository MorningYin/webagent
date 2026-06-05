---
name: segment-sft-pipeline
description: WebGym 4-stage offline pipeline that builds segment-level credit-assignment SFT data for long-horizon web-agent RL
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

WebGym builds SEGMENT-LEVEL credit-assignment SFT data for long-horizon web-agent RL. Research insight: a single action is too fine to credit and a whole trajectory is too coarse, so trajectories are cut into "skill segments" (sub-goal units) — humans learn skills, not isolated tool calls. The offline pipeline has 4 stages, all driven by committed scripts in `/root/webgym/scripts/` (branch `segment-sft-pipeline`):

- Stage 1 — Data collection: rollout of gpt-5.4 (closed API) produces trajectories. `scripts/export_final.py` consolidates run_v3 trajectories into a training-ready bundle; `scripts/rebuild_remaining.py` is the resume-safe task-list rebuilder (never re-runs already-collected tasks). Output landed as the clean dataset — see [[clean-dataset-artifacts]].
- Stage 2 — Segment division: `scripts/segmenter.py` uses gpt-5.5 (dkyx gateway only) to relabel each flattened trajectory into contiguous head-to-tail skill segments, each with four descriptive fields (observation / purpose / summary / result). Details in [[segmenter-design]].
- Stage 3 — Within-segment summary: `scripts/build_seg_summaries.py` uses gpt-5.4-mini (yunwu only) to build per-step rolling notes (two aligned versions). Details in [[within-segment-summary-design]].
- Stage 4 — Data construction: `scripts/build_master_with_segments.py` reconstructs the lossless master message array and inlines `segment_beginner` / `segment_stopper` meta-calls at segment boundaries, adding a detailed segment spec to the system prompt. Details in [[message-array-and-token-budget]].

Downstream (on a different machine) the final per-step training input is assembled from the master message array + within-segment summary + screenshots, using a three-tier bounded memory: A = history of closed segments, B = within-segment summary, plus a sliding image window. Data correctness is enforced by RL + the segment-level critic, NOT by trajectory success rate (~1.6%) — see [[sft-purpose-vs-rl-reward]]. Serving/infra (LiteLLM proxy, gateways, sealed proxy chain) in [[webgym-closed-api-rollout-setup]]; packaging and git/branch state in [[data-migration-and-git-state]].

**Why:** segment-level credit assignment is the core research bet; knowing the 4 stages, their exact scripts, and which model/gateway each stage uses lets future work extend or debug the pipeline without re-deriving it, and prevents conflating SFT-data construction with trajectory success rate.
**How to apply:** when touching the SFT data pipeline, start from the matching stage script in `/root/webgym/scripts/`, respect the per-stage model/gateway pinning (segmenter→gpt-5.5/dkyx, summaries→gpt-5.4-mini/yunwu), keep the master message array lossless, and follow the cross-linked memories for each stage's design specifics.
