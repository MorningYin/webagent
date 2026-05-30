#!/usr/bin/env bash
# Cleanly stop the rollout client AND its process-pool workers.
# rollout.py spawns ~hundreds of worker processes; killing only the main PID
# orphans them (they keep burning CPU and pile up across restarts).
# Uses a /proc scan that excludes the current process tree (a pkill/-f loop over
# the same pattern can signal this shell -> exit 144), so it kills reliably.
python3 - <<'PY'
import os, signal, time
me = os.getpid(); safe = {me, os.getppid(), 1}
def sweep():
    k = 0
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in safe:
            continue
        try:
            cl = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode('utf8', 'ignore')
        except Exception:
            continue
        if 'rollout.py' in cl:
            try: os.kill(int(d), signal.SIGTERM); k += 1
            except Exception: pass
    return k
n = sweep(); print(f"SIGTERM sent to {n} rollout procs"); time.sleep(5)
# force-kill survivors
k = 0
for d in os.listdir('/proc'):
    if not d.isdigit() or int(d) in safe: continue
    try: cl = open(f'/proc/{d}/cmdline','rb').read().replace(b'\0',b' ').decode('utf8','ignore')
    except Exception: continue
    if 'rollout.py' in cl:
        try: os.kill(int(d), signal.SIGKILL); k += 1
        except Exception: pass
print(f"SIGKILL'd {k} survivors")
PY
echo "rollout stopped."
