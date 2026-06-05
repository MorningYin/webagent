#!/usr/bin/env python
"""Consolidate ALL run_v3 trajectories and export a training-ready + summary bundle.
Run anytime (e.g. when the relay quota runs out) to snapshot the current state.

Outputs to /root/autodl-tmp/webgym_runs/export/final/:
  final_concluded.pt            concluded-only trajectories (high quality, for SFT/RL)
  final_concluded_steps_sft.jsonl  per-step raw SFT pairs {prompt(raw_prompt), response(raw_response)}
  final_all.pt                  ALL unique trajectories deduped (incl truncated/garbage; for DPO negatives)
  final_summary.txt             stats: counts, reward dist, by-difficulty, step distribution
"""
import torch, glob, os, json, re, collections, statistics

SRC_GLOB = "/root/autodl-tmp/webgym_runs/run_v3/train_trajectories/*"
OUT = "/root/autodl-tmp/webgym_runs/export/final"
os.makedirs(OUT, exist_ok=True)

def lastact(t):
    tc=(getattr(t[-1].get('response'),'answering_tokens',{}) or {}).get('tool_call') or {}
    return (tc.get('arguments') or {}).get('action') if isinstance(tc,dict) else None
def concluded(t):
    if len(t)<2: return False
    r=t[-1].get('reward'); ev=str(getattr(r,'evaluation','') or '')
    return (getattr(r,'submit',None) is True) or lastact(t)=='answer' or 'Criterion' in ev or 'Verdict' in ev
def tid(t):
    try: return str(t[0]['observation'].task.task_id)
    except: return None
def difficulty(t):
    try: return getattr(t[0]['observation'].task,'difficulty',None)
    except: return None

# consolidate: per task_id keep best (concluded > truncated > garbage)
best={}
for f in sorted(glob.glob(SRC_GLOB)):
    if os.path.isdir(f): continue
    try: d=torch.load(f, weights_only=False)
    except Exception as e: print(f"skip {os.path.basename(f)}: {e}"); continue
    ts=d['trajectories'] if isinstance(d,dict) else d
    for t in ts:
        i=tid(t)
        if not i: continue
        rank=2 if concluded(t) else (1 if len(t)>=2 else 0)
        if i not in best or rank>best[i][0]: best[i]=(rank,t)

allt=[t for _,t in best.values()]
conc=[t for r,t in best.values() if r==2]
trunc=[t for r,t in best.values() if r==1]
garb=[t for r,t in best.values() if r==0]

torch.save({'trajectories':conc,'metadata':{'kind':'concluded_only','judge':'gpt-5.4'}}, f"{OUT}/final_concluded.pt")
torch.save({'trajectories':allt,'metadata':{'kind':'all_unique_deduped'}}, f"{OUT}/final_all.pt")

# per-step SFT pairs (concluded only, raw)
nstep=0
with open(f"{OUT}/final_concluded_steps_sft.jsonl","w") as fo:
    for ti,t in enumerate(conc):
        i=tid(t); rew=getattr(t[-1].get('reward'),'reward',None)
        for si,s in enumerate(t):
            resp=s.get('response'); rp=getattr(resp,'raw_prompt',None); rr=getattr(resp,'raw_response',None)
            if not rp or not rr: continue
            fo.write(json.dumps({"traj_idx":ti,"task_id":i,"step":si,"prompt":rp,"response":rr,"traj_reward":rew},ensure_ascii=False)+"\n")
            nstep+=1

# summary stats
rew=collections.Counter(getattr(t[-1].get('reward'),'reward',None) for t in conc)
diff=collections.Counter(difficulty(t) for t in conc)
steps=[len(t) for t in conc]
with open(f"{OUT}/final_summary.txt","w") as f:
    f.write("="*70+"\nWebGym run_v3 final export summary\n"+"="*70+"\n")
    f.write(f"unique tasks attempted: {len(best)}\n")
    f.write(f"  CONCLUDED (kept, high-quality): {len(conc)}\n")
    f.write(f"  truncated (ran out, not kept):  {len(trunc)}\n")
    f.write(f"  garbage (<2 steps):             {len(garb)}\n\n")
    f.write(f"concluded reward distribution: {dict(rew)}  (reward==1: {rew.get(1,0)}, {rew.get(1,0)/max(len(conc),1)*100:.1f}%)\n")
    f.write(f"concluded by difficulty: {dict(diff)}\n")
    if steps:
        f.write(f"concluded steps: median {statistics.median(steps)}, mean {sum(steps)/len(steps):.1f}, max {max(steps)}\n")
    f.write(f"\nper-step SFT pairs (concluded): {nstep}\n")
    f.write(f"\nfiles in {OUT}/:\n  final_concluded.pt ({len(conc)} traj)\n  final_concluded_steps_sft.jsonl ({nstep} pairs)\n  final_all.pt ({len(allt)} traj)\n")
    f.write("NOTE: images referenced as file:// under run_v3/images/ — ship that dir alongside.\n")

print(open(f"{OUT}/final_summary.txt").read())
print(f"WROTE bundle to {OUT}/  | sft pairs: {nstep}")
