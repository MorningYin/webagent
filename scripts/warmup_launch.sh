#!/usr/bin/env bash
# 发车前热身:把 sealed 代理(7891)的静态IP隧道预热 + 确认稳定,再启动 rollout。
# 防止 36 个浏览器冷启动惊群撞上冷/抖动的代理 -> step-0 大面积空白。
# 用法: bash scripts/warmup_launch.sh
set -u
PROXY=http://127.0.0.1:7891
SITES=(https://archive.org https://en.wikipedia.org https://www.google.com https://www.bing.com https://amazon.com)

warm_round() {  # 并发打 N 个,回成功率%
  local N=$1; local ok=0
  for j in $(seq 1 $N); do
    u=${SITES[$((RANDOM % ${#SITES[@]}))]}
    ( c=$(env -u http_proxy -u https_proxy -u all_proxy curl -s -o /dev/null -w '%{http_code}' --max-time 15 -x $PROXY "$u" 2>/dev/null); [ "$c" != "000" ]&&[ -n "$c" ] && echo ok >> /tmp/warm_$$.tmp ) &
  done
  wait
  ok=$(wc -l < /tmp/warm_$$.tmp 2>/dev/null || echo 0); rm -f /tmp/warm_$$.tmp
  echo $((ok*100/N))
}

echo "=== 代理热身 + 稳定性闸门 ==="
good=0
for r in $(seq 1 8); do
  rate=$(warm_round 30)
  echo "  热身轮$r: 30并发成功率 ${rate}%"
  if [ "$rate" -ge 90 ]; then good=$((good+1)); else good=0; fi
  [ "$good" -ge 2 ] && { echo "✅ 代理已热且稳定(连续2轮≥90%),放行发车"; break; }
  sleep 4
done
if [ "$good" -lt 2 ]; then echo "⚠️ 代理未达稳定(可能静态IP在抖),仍发车但可能有少量 burst"; fi

echo "=== 启动 rollout ==="
cd /root/webgym/scripts
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/.cache/ms-playwright \
    POLICY_PROXY_URL="http://localhost:4000/v1/chat/completions" POLICY_PROXY_BASE="http://localhost:4000/v1" \
    LITELLM_MASTER_KEY="sk-local-webgym-pool" CPU_CLUSTER_TOKEN="default_key" WANDB_MODE=disabled WANDB_DISABLED=true \
    nohup /root/miniconda3/envs/webgym/bin/python rollout.py --config-name rollout_5k \
    > /root/autodl-tmp/run_v3.log 2>&1 &
echo "rollout pid $!"
