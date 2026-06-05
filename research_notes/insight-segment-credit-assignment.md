---
name: insight-segment-credit-assignment
description: "The unit of credit assignment for long-horizon agentic RL is the skill segment — not the action, not the trajectory"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

This is the central research thesis of WebGym — the reason the whole project exists.

**The claim:** For long-horizon agentic RL, the correct unit of credit assignment is the SKILL SEGMENT, not the individual action and not the whole trajectory.

- A single ACTION cannot be credited: there is no local signal telling you whether one click was "good." Per-action RL is too noisy / unsignalled.
- A whole TRAJECTORY is too coarse: one scalar reward spread over 30–400 steps cannot say WHICH part of the run actually worked. Per-trajectory RL throws away all the within-episode structure.
- A SEGMENT — a contiguous run of steps pursuing one sub-goal / one state-transition attempt — is the smallest unit that is locally judgeable. It is the sweet spot strictly between per-action (too fine) and per-trajectory (too coarse).

**Human-learning analogy (the motivating intuition):** People do not learn the goodness of individual motor actions. They learn SKILLS — reusable combinations of actions that have a purpose and an observable outcome. An agent should likewise learn skills, not isolated tool calls. The segment is the computational stand-in for "a skill."

**The contribution (paper framing):** Cut trajectories into skill segments, then perform credit assignment / critic scoring at the segment level — each segment scored as (before-state → actions → after-state) for one skill. This reframing is the novel move; it is what lets credit flow to the part of a long episode that mattered.

**Consequence for the data pipeline:** The offline SFT pipeline exists to TEACH this segment structure — the policy itself learns to emit segment boundaries (see [[insight-policy-intrinsic-segmentation]], [[segment-sft-pipeline]]). Correctness is then enforced by RL + a segment-level critic, NOT by trajectory success rate. Trajectory success (~1.6%) is explicitly NOT a quality gate (see [[sft-purpose-vs-rl-reward]]). The bounded-context harness (see [[insight-three-tier-bounded-memory]]) keeps each segment's state-transition view tractable over very long horizons.

**Why:** Credit assignment fails at both extremes — actions carry no local reward signal, trajectories carry one signal too diffuse to localize; the skill segment is the smallest locally-judgeable unit, mirroring how humans learn reusable skills rather than isolated motions.

**How to apply:** Frame all WebGym method/critic/RL work around the segment as the atomic unit of credit; segment trajectories into skill spans, score each as before→actions→after, and never treat trajectory success rate as a quality gate — it is a teaching signal for structure, not correctness.
