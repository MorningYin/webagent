---
name: data-migration-and-git-state
description: "Moving the segment-aware dataset bundle server->local->target via rsync, plus webgym git branch/remote/secrets/cleanup state"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

Two things: (A) the dataset migration bundle, (B) the git repo state and cleanup. See also [[segment-sft-pipeline]] (the code that produced the data), [[clean-dataset-artifacts]] (the layout inside the bundle), and [[webgym-closed-api-rollout-setup]] (the API keys now in .env).

## (A) Migration bundle (server -> local -> target)
The segment-aware dataset must reach ANOTHER machine for downstream assembly. This AutoDL box CANNOT reach the target; only the user's local laptop can reach both, so the flow is **server -> local -> target** (two hops).

Bundle files in `/root/autodl-tmp/webgym_runs/export/`:
- `webgym_segdata.tar.zst` — 9.9G (from 16G, zstd -3). Regeneratable from `clean_dataset/`; can be deleted to reclaim space.
- `webgym_segdata.tar.zst.sha256` — PORTABLE bare-filename checksum (`b3820d22...3f56  webgym_segdata.tar.zst`) so `sha256sum -c` works from any directory.
- `TRANSFER_GUIDE.md` — the how-to.

Bundle contents (rooted to restore the `clean_dataset/` layout on extract): `MANIFEST.txt` + `messages_with_segments.jsonl` + `segments.jsonl` + `seg_histories_dual.jsonl` + `images/`. These are the 4 things the user needs: message array + four-field segments + within-segment summary + screenshots.

Transfer = rsync over SSH with resume, two hops via local.
- macOS gotcha: `which rsync` hits the old `/usr/bin` openrsync 2.6.9 (no `--append-verify`). Use brew rsync: `RSYNC="$(brew --prefix rsync)/bin/rsync"` (or fix PATH). Old-rsync fallback = `--partial --append`.
- BOTH ends need rsync installed (this server has 3.2.7).
- Extract on target (needs zstd; this box has zstd 1.5.5): `zstd -dc webgym_segdata.tar.zst | tar -xf -`.

## (B) Git state & cleanup
Code lives in `/root/webgym` (git repo, user MorningYin). Pipeline work committed on branch **`segment-sft-pipeline`** (commit `4d8d331` "Segment-level SFT data pipeline + closed-API rollout hardening") and PUSHED to remote **`webagent` = git@github.com:MorningYin/webagent.git** via SSH.

IMPORTANT: remote **`origin` points to https://github.com/microsoft/webgym.git** (the upstream — do NOT push project work there). Push only to `webagent`.

SSH: an ed25519 key (`~/.ssh/id_ed25519`, comment "webgym-autodl") was generated on this AutoDL box and added to GitHub; `github.com:22` is reachable directly (no proxy needed for SSH).

SECRETS: 5 relay API keys were moved OUT of `scripts/start_litellm.sh` into `/root/webgym/.env` (gitignored; `git check-ignore .env` confirms) with `.env.example` as the committed template; `start_litellm.sh` now sources `.env`. The keys were NEVER committed to git history (verified). `.gitignore` also ignores `.ipynb_checkpoints/` and `data/` (chroma_db).

Repo cleanup: deleted root-level dead duplicates `segmenter.py` + `critic.py`, and superseded helpers `scripts/build_master_messages.py`, `estimate_tokens.py`, `export_trajectories.py` (backed up to `/tmp/webgym_scripts_cut/` during the session). The 5 KEPT pipeline scripts: `segmenter.py`, `build_seg_summaries.py`, `build_master_with_segments.py`, `export_final.py`, `rebuild_remaining.py`.

CLONE caveats on a new machine:
- `.env` is NOT in the repo — `cp .env.example .env` and fill keys.
- scripts hardcode `/root/autodl-tmp/...` paths — adjust.
- the 16G data is NOT in git — transfer via the tar bundle above.

**Why:** the dataset only lives on this unreachable AutoDL box, and the repo's `origin` is the Microsoft upstream — both are easy to get wrong (push to the wrong remote, or lose the regeneratable 9.9G bundle / its SSH transfer recipe) without these specifics.
**How to apply:** to ship data, rsync `webgym_segdata.tar.zst`(+`.sha256`) server->local->target with rsync 3.2.7 (brew on macOS), verify with `sha256sum -c`, extract via `zstd -dc ... | tar -xf -`; to push code, target remote `webagent` (never `origin`); to redeploy, clone `webagent`, `cp .env.example .env` + fill keys, fix hardcoded paths.
