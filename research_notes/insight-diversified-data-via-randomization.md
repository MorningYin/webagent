---
name: insight-diversified-data-via-randomization
description: "Deliberately build diversified training data via randomized image-window k + dual rolling representations; materialize raw aligned materials, defer sample realization to downstream assembler"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

Design stance for the segment SFT data: do NOT bake one fixed memory/summary shape into the materialized dataset. Instead produce raw, aligned, randomizable materials at construction time and let the downstream assembler realize many concrete training variants from them. See [[segment-sft-pipeline]], [[within-segment-summary-design]], [[insight-three-tier-bounded-memory]], [[message-array-and-token-budget]].

**Randomized sliding-window k (image vs text split).**
When assembling each per-step training sample, the within-segment image window size `k` is sampled from {1,2,3,4,5} (k_min=1). For a step at within-segment position `j`, the recent `k` steps stay as images, and the within-segment text summary B becomes the prefix concatenation of per-step delta notes for the steps that fall OUTSIDE that window:
  `B = join(rolling[0 : j-k])`.
So the SAME step yields different (how-much-is-image vs how-much-is-text) splits depending on the sampled `k`. The stopper step counts as a within-segment step (the deepest lookup), so even an L=2 segment still produces a prefix-summary step.

**Two aligned rolling representations kept on purpose.**
Each segment stores BOTH, same length, per step:
- `rolling_structured` — deterministic event line: `from → action → to ; surface ; thought`.
- `rolling_model` — gpt-5.4-mini natural-language one-sentence note.
They are retained deliberately so downstream can build diversified data: mix deterministic vs natural-language memory phrasings, or ablate which representation trains the policy better — rather than committing to one summary style up front.

**Why randomize k:** it diversifies the training distribution over "how much recent visual context vs how much summarized text" the policy must operate on, making the learned policy robust to different memory cutoffs at inference — it can act from mostly-text memory OR mostly-image recent context, not just one fixed window.

**Why keep dual rolling versions / raw materials:** one materialized dataset can spawn many training variants (different k, different rolling style, different count of history segments in A) without re-running the expensive segmentation/summarization. Raw aligned materials + randomizable cut points = a recurring leverage point in this project.

**How to apply:** at data-construction time, emit multiple aligned representations and randomizable cut points (windows, prefixes, history depth); defer the actual sample realization (which k, which rolling version, how many history segments in A) to the downstream assembler. When adding a new memory/context knob, prefer making it a sampled/configurable dimension over hardcoding one value, so it becomes an ablation axis for free.
