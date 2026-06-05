---
name: insight-weak-model-as-data-tool
description: "When a cheap model (gpt-5.4-mini) is usable (format/SFT-bootstrap, structured rewrite) vs not (correctness-critical generation)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

A cheap/weak model (gpt-5.4-mini) can be used for FORMAT and SFT-bootstrap-style tasks and for cheap STRUCTURED REWRITING of already-correct content, but NOT for correctness-critical generation.

EMPIRICAL A/B (gpt-5.4-mini as a cheap rollout/data source):
- With a mini-SPECIFIC prompt (strong format rules + few-shot), format adherence was fixed: unparseable dropped to ~16.9% (vs ~100% unparseable with the plain prompt), and ~68% reached a conclusion.
- BUT correctness stayed weak: reward==1 only ~21% vs ~54% for gpt-5.4.
- Conclusion: mini is viable for format / SFT-bootstrap tasks, not for correctness.

DESIGN PRINCIPLE (capability shapes architecture): put the STRONG model where correctness matters (rollout policy = gpt-5.4; segmenter = gpt-5.5), and use the WEAK/cheap model only for cheap structured rewriting of already-correct content (within-segment per-step summaries = gpt-5.4-mini).

CONCRETE CONSEQUENCE in the summary pipeline: the within-segment summary uses "DESIGN B" — per-step DELTA notes concatenated downstream — specifically BECAUSE mini is too weak to write cumulative, self-contained running summaries. Each note describes only its own event; the cumulative memory is assembled later by concatenation, not by the model. (See [[within-segment-summary-design]].)

RELATED FAILURE MODE (don't let the weak model — or over-engineering — do extraction): trying to make mini's input cheaper via regex fact-extraction from `thought` was a REGRESSION (garbage facts, dropped real values). Fix: feed mini the FULL thought and let it do only the light rewrite. Weak model + deterministic preprocessing is fine; weak model fed lossy/over-engineered input is worse than feeding it the raw text.

**Why:** Format adherence and faithful local rewriting are low-capability tasks a cheap model can do reliably, but correctness/cumulative reasoning is a high-capability task where the cheap model collapses (21% vs 54% reward) — so model capability, not cost alone, must dictate where each model sits in the pipeline.

**How to apply:** Route by task type — strong model for any step whose output's correctness propagates (policy rollouts, segmentation), cheap model only for per-step/structured rewriting of content already known to be correct; never make the cheap model produce cumulative state or do lossy preprocessing of its own input — give it the full raw text and a tight format spec.

Cross-links: [[within-segment-summary-design]], [[segment-sft-pipeline]], [[segmenter-design]], [[webgym-closed-api-rollout-setup]]
