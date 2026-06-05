---
name: insight-policy-intrinsic-segmentation
description: "Segmentation reframed as two policy-emitted meta-calls (segment_beginner/stopper); SFT teaches the format, RL+critic grows skill, causal no-hindsight fields keep train==inference"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

The meta-call design turns trajectory segmentation from an external bolt-on into something the policy itself produces. Four conceptual claims, all load-bearing:

1. **From external annotation to policy-intrinsic.** Segmentation is no longer "an annotator post-hoc cuts the trajectory." It is reframed as TWO TOOLS THE POLICY ITSELF CALLS during rollout: `segment_beginner(observation, purpose)` OPENS a segment (one contiguous run of steps pursuing a single objective / state-transition) and `segment_stopper(summary, result)` CLOSES it. Their natural home is the policy emitting them INLINE while it acts — not a user-side marker. The segment structure becomes a thing the model generates, not something attached after the fact. (A length-1 segment carries BOTH meta-calls around its single Thought/Action/computer_use turn; middle steps carry neither.)

2. **SFT bootstrap teaches the FORMAT, not correctness.** Collected trajectories are REWRITTEN so segment boundaries become ASSISTANT `tool_calls` — inline `<tool_call>{"name":"segment_beginner",...}</tool_call>` prepended to / appended to the boundary step's assistant turn — NOT user-side XML markers. This is the critical detail: if the boundaries lived on the user side, the policy would never learn to EMIT them itself. Confirmed in `/root/webgym/scripts/build_master_with_segments.py`: `build_one()` (lines ~137-146) prepends the beginner `tool_call` and appends the stopper `tool_call` into `resp[step]`, which is then written as the `{"role":"assistant","content":...}` message (line ~156); the two meta-tool specs are also injected into the system `<tools>` block. SFT's job is to teach the format + the HABIT of emitting the two meta-calls; it is bootstrap, not the source of correctness.

3. **RL + critic is where skills emerge.** Online, the policy emits beginner/stopper itself; a segment-level critic reads the four-tuple (observation / purpose / summary / result = before-state / intent / process / after-state) and scores the segment. Skill quality comes from RL on those segment-level signals, NOT from imitation. This is exactly why SFT trajectory-success / correctness is not a gate (the ~1.6% webgym success rate is irrelevant as a quality bar — see [[sft-purpose-vs-rl-reward]]).

4. **The causal / no-hindsight constraint (easy to get wrong, must hold).** Field dependencies must be respected so the SFT target is causally imitable:
   - `beginner = f(memory-so-far A + current screen) → (observation, purpose)`. These may use ONLY information available BEFORE the segment acts — NO hindsight, no describing what will later be discovered. If the target leaks the future, the policy cannot reproduce it at inference.
   - `stopper = f(beginner + actions actually taken) → (summary, result)`.
   The segmenter that writes these four fields (see [[segmenter-design]]) must be strongly constrained to honor this (observation/purpose strictly pre-action). This buys train==inference consistency: at inference the policy emits `beginner` from only A + screen, exactly as trained. All four fields are descriptive records, not quality/reward judgements.

   *Minor acknowledged train/SFT mismatch:* the old per-step Thought/Action text was generated under the OLD free-form running_log; rewriting the prompt to the new A/B three-tier structure (see [[insight-three-tier-bounded-memory]], [[message-array-and-token-budget]]) causes slight wording mismatch (same-source content, but the model must learn to read the new format). Acceptable for SFT bootstrap; it vanishes in RL once the policy self-produces the log.

Cross-links: [[segment-sft-pipeline]], [[insight-segment-credit-assignment]], [[insight-three-tier-bounded-memory]], [[segmenter-design]], [[message-array-and-token-budget]], [[sft-purpose-vs-rl-reward]].

**Why:** Making segmentation a policy-emitted action (vs. external annotation) is what lets RL+critic shape per-segment skill at all; the no-hindsight field constraint is what makes the SFT target causally imitable so train and inference match — get either wrong and the whole credit-assignment scheme silently breaks.
**How to apply:** When editing the SFT rewrite or segmenter, keep beginner/stopper as ASSISTANT `tool_calls` (never user markers), keep observation/purpose strictly pre-action (no future leakage), treat SFT as format-bootstrap only, and rely on the segment-level critic + RL — not trajectory success — for correctness.
