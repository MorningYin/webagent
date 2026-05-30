# SETUP_CONTEXT — 在新机器上部署本项目（闭源 API 策略收集 web rollout）

> 这份文档让你在一台**全新机器**上从零复现整套系统。读完按步骤走即可。
> 仓库：`https://github.com/MorningYin/webagent.git`（fork 自 microsoft/webgym + 本项目改动）。

---

## 0. 这个项目在干嘛（一句话）

用 **WebGym** 框架，让一个**闭源大模型 API（gpt-5.4，经中转站）当策略**，在真实浏览器（OmniBoxes 控制的 Chromium）里跑 web 研究任务，**收集大量多步轨迹**。每步发给模型的 **OpenAI 格式 message 数组**（含截图）会完整存进 `.pt` 文件，用作训练数据。

- **不需要 GPU**：推理全在云端 API，本机只跑浏览器（吃 CPU）。
- **不训练**：只用 rollout 这一半（README 说 WebGym 的 rollout 部分可独立抽出）。

---

## 1. 相对上游 WebGym 改了什么（本项目的全部改动）

| 文件 | 改动 |
|---|---|
| `webgym/models/web_agent.py` | **新增闭源 API 策略后端**：当 `policy_config.api.base_url` 配了就走 OpenAI 兼容 API（否则保持原 vLLM 路径不变）。含：图片 `file://`→base64 内联（远程 API 读不到本地盘）；**限流加固重试**（429/5xx/网络错/空-200 都重试，尊重 `Retry-After`，指数退避+抖动+封顶）。 |
| `omniboxes/node/instances/playwright_instance.py` | 浏览器启动读 `OMNIBOX_BROWSER_PROXY` 环境变量（留空=直连）。 |
| `webgym/environment/async_webgym.py` | 监控面板端口可配，默认 **6006**（AutoDL 自定义服务正好代理 6006）；修了 `stack_size` 报错。 |
| `scripts/config/main/rollout_smoke.yaml` | 冒烟配置（3 任务，验证管线）。 |
| `scripts/config/main/rollout_5k.yaml` | 生产配置（5000 条，可断点续跑）。 |
| `scripts/config/main/rollout.yaml` | 加了注释版 API 配置范例。 |
| `scripts/watch_rollout.py` | 命令行实时监控（进度 + 已完成轨迹内容摘要）。 |

---

## 2. 关键架构（必须理解，否则配不通）

```
            ┌─────────────────────────── 本机 ───────────────────────────┐
任务 → rollout 客户端(scripts/rollout.py)
          │  ① 策略推理：把 OpenAI messages 直接 POST 到中转站 → gpt-5.4
          │     （国内直连中转站，不走代理）
          │  ② 浏览器操作：通过 OmniBoxes master(:7000) 控制 N 个 Chromium
          ▼
       OmniBoxes 浏览器集群(:7000→:8080→:9000+)，每个 Chromium 访问真实网站
          │  浏览器出口 → 密封代理(:7891)
          ▼
    密封 mihomo(:7891)  ──机场翻墙(第一跳)──→ 5 个静态IP(第二跳,稳定出口/防封) ──→ 目标网站
```

**两个模型角色**（都走中转站，OpenAI 兼容）：
- **策略 policy**：gpt-5.4（产生动作）。
- **评判 evaluator**：gpt-5.4-mini（算 reward / 检测被拦），rollout 主循环必须有它，不能留空。

**为什么浏览器要两跳代理**：那 5 个静态 IP **拒绝中国大陆来源**（实测报 `Mainland China IP ... banned`）。所以必须先经机场翻墙变成海外源 IP，再连静态 IP，由静态 IP 做稳定出口（5 个轮询分摊并发、防封）。

---

## 3. 部署步骤

### 步骤 1：拉代码 + 建隔离 conda 环境 + 装依赖

```bash
git clone https://github.com/MorningYin/webagent.git /root/webgym
cd /root/webgym

conda create -y -n webgym python=3.11
PIP=/root/miniconda3/envs/webgym/bin/pip
PY=/root/miniconda3/envs/webgym/bin/python

# torch 用 CPU 版即可（API 策略不需要 GPU；torch 只用于存 .pt / 设种子）
$PIP install torch --index-url https://download.pytorch.org/whl/cpu

# 依赖（去掉 vllm/gradio/gym 等重包；gradio 仅内容查看器需要，可选）
$PIP install "fastapi[standard]" httpx "Pillow==11.2.1" playwright redis "Requests==2.32.3" \
  "psutil==7.0.0" "transformers==4.57.1" "hydra-core==1.3.2" "omegaconf==2.3.0" pandas numpy \
  tenacity wandb nltk tqdm matplotlib fastparquet "antlr4-python3-runtime==4.9.3" \
  ipython ipywidgets openai huggingface_hub flask
$PIP install gradio          # 可选：仅 analysis/view_trajs.py 内容查看器需要

# 关键：让 import webgym 解析到本仓库（否则可能解析到别处的旧包）
$PIP install -e /root/webgym --no-deps

# Playwright 浏览器（若已存在可设 PLAYWRIGHT_BROWSERS_PATH 复用）
$PY -m playwright install chromium
```

> 验证：`$PY -c "import webgym,os;print(os.path.dirname(webgym.__file__))"` 应指向本仓库。

### 步骤 2：redis（OmniBoxes 状态协调）

```bash
redis-server --port 6379 --daemonize yes --save '' --appendonly no
redis-cli ping   # 应返回 PONG
```

### 步骤 3：密封浏览器代理（两跳链，端口 7891）

前提：本机已有一个 mihomo（你自用的机场订阅，含机场节点）。我们**复制它的机场节点**、排除你自用静态节点、加上 5 个静态 IP 做 load-balance，**另起一个独立实例**（独立端口/数据目录，不碰你自用代理）。

> ⚠️ 下面的 5 个静态 IP（host:port:user:pass）和机场订阅是**你的私密信息，不在仓库里**。在新机器上填你自己的值。

把下面脚本存成 `gen_omnibox_proxy.py`，改好 `statics` 列表里的 5 个静态 IP，然后运行：

```python
import yaml
SRC = "/root/.config/mihomo/config.yaml"        # 你自用 mihomo 的配置
PERSONAL_NODE_KEYWORD = "AI静态IP"               # 要排除的自用节点名关键字
statics = [   # ← 填你自己的 5 个静态 IP（host, user, pass）
 ("static-1","<IP1>","<user1>","<pass1>"),
 ("static-2","<IP2>","<user1>","<pass1>"),
 ("static-3","<IP3>","<user1>","<pass1>"),
 ("static-4","<IP4>","<user1>","<pass1>"),
 ("static-5","<IP5>","<user5>","<pass5>"),
]
cfg = yaml.safe_load(open(SRC))
jichang = [p for p in cfg["proxies"] if PERSONAL_NODE_KEYWORD not in p["name"]]
jnames = [p["name"] for p in jichang]
sp = [{"name":n,"type":"http","server":s,"port":443,"username":u,"password":pw,
       "tls":False,"dialer-proxy":"jichang-auto"} for (n,s,u,pw) in statics]
snames=[n for (n,*_) in statics]
out={"mixed-port":7891,"allow-lan":False,"mode":"rule","log-level":"warning",
 "external-controller":"127.0.0.1:9098","proxies":jichang+sp,
 "proxy-groups":[
   {"name":"jichang-auto","type":"url-test","proxies":jnames,
    "url":"http://cp.cloudflare.com/generate_204","interval":1800,"tolerance":150},
   {"name":"static-lb","type":"load-balance","strategy":"round-robin","proxies":snames,
    "url":"http://cp.cloudflare.com/generate_204","interval":600}],
 "rules":["MATCH,static-lb"]}
import os; os.makedirs("/root/.config/mihomo-omnibox",exist_ok=True)
yaml.safe_dump(out,open("/root/.config/mihomo-omnibox/config.yaml","w"),allow_unicode=True,sort_keys=False)
print("wrote /root/.config/mihomo-omnibox/config.yaml; 机场",len(jnames),"静态",snames)
```

启动并验证（应轮询出 5 个静态 IP）：

```bash
nohup mihomo -d /root/.config/mihomo-omnibox > /root/.config/mihomo-omnibox/run.log 2>&1 &
for i in 1 2 3 4 5; do curl -s -x http://127.0.0.1:7891 https://api.ipify.org; echo; done
```

> 若新机器只有一个 mihomo 端口，确保密封实例的 `mixed-port`(7891) 和 `external-controller`(9098) 与自用实例不冲突。

### 步骤 4：密钥与环境变量（**不要写进仓库**）

```bash
export POLICY_API_KEY="<你的 yunwu/new-api key>"   # 策略+评判共用
export CPU_CLUSTER_TOKEN="default_key"              # OmniBoxes master 的 api-key（代码里写死 default_key）
```

中转站（OpenAI 兼容，都服务 gpt-5.4）：
- **yunwu**：`base_url = https://yunwu.ai/v1`，基本不限流。
- **new-api**：另一个站（更便宜，可能限流），base_url 用你那个站的地址。

### 步骤 5：改配置 `scripts/config/main/rollout_5k.yaml`

需要按新机器改的字段：
- `save_path`：轨迹/截图输出目录（放**大磁盘**，5k 约 18–30GB）。
- `data_path`：含 `train.jsonl` 的目录（HF 数据集 `microsoft/webgym_tasks`，首次会自动下载到 HF 缓存）。
- `policy_config.api.base_url` / `model`：选 yunwu 或 new-api / gpt-5.4。
- `openai_config`（评判）：`base_url` + `model: gpt-5.4-mini` + `openai_api_key_env_var: POLICY_API_KEY`。
- `env_config.server_size` / `max_vllm_sessions`：并发（见第 5 节调优）。

### 步骤 6：启动 OmniBoxes（带密封代理、剥离自用代理 env）

`N` = 并发浏览器数，要 ≥ `server_size`。

```bash
cd /root/webgym/omniboxes/deploy
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    OMNIBOX_BROWSER_PROXY=http://127.0.0.1:7891 \
    nohup /root/miniconda3/envs/webgym/bin/python deploy.py 24 --master-port 7000 \
    > /root/autodl-tmp/omnibox_deploy.log 2>&1 &
# 健康检查（capacity 应等于 N）
curl -s -H "x-api-key: default_key" http://localhost:7000/info
```

### 步骤 7：跑 rollout（隔离环境，clear 代理→API 直连中转站+本地）

```bash
cd /root/webgym/scripts
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    POLICY_API_KEY="$POLICY_API_KEY" CPU_CLUSTER_TOKEN="default_key" \
    WANDB_MODE=disabled WANDB_DISABLED=true \
    nohup /root/miniconda3/envs/webgym/bin/python rollout.py --config-name rollout_5k \
    > /root/autodl-tmp/run5k.log 2>&1 &
```
断点续跑：`save_traj_progress: true`，崩了重发同条命令会接着存。停止：`kill <rollout_pid>`（别 kill -9，避免 OmniBoxes lease 泄漏）。

---

## 4. 监控（3 个入口）

1. **网页面板**：跑起来自动开在 `:6006`。AutoDL→自定义服务开 6006；或 `ssh -L 6006:localhost:6006`。看完成数/每任务步数与状态。
2. **命令行**（最稳，无需端口）：
   ```bash
   /root/miniconda3/envs/webgym/bin/python /root/webgym/scripts/watch_rollout.py \
     --save-path <save_path> --content 10
   ```
3. **内容查看器**（截图+每步 prompt/response，需 gradio）：
   ```bash
   /root/miniconda3/envs/webgym/bin/python analysis/view_trajs.py train \
     --data-path <data_path> --log-path <save_path> --show-prompt
   ```

---

## 5. 调优与坑（用代价换来的经验）

- **CPU 是瓶颈，不是模型/代理**：每个 Chromium 渲染真实网页很吃 CPU。实测 16 核机器上 **24 浏览器 ≈ load 10–15**，已接近上限。提并发前先看 `uptime` 的 load；load 持续 ≥核数 就会出 503/超时/浏览器崩。**真要提速 → 加 CPU 核 / OmniBoxes 多节点**（横向扩），不是在单机硬堆并发。
- **`server_size` / `max_vllm_sessions` / `deploy.py N` 三者要一起调**，且 `N ≥ server_size`。
- **速度根因**：gpt-5.4 带 5 图+推理，单步 ~30–40s，任务最多 30 步 → 单条 ~18min。单机 24 并发 ≈ 1.3 轨迹/分，5k ≈ 60+ 小时。想快：加核/多节点，或降 `*_difficulty_max_steps`（数据质量权衡）。
- **静态 IP 拒绝大陆来源** → 浏览器代理**必须**两跳（机场→静态IP）。直接拿静态 IP 当代理会 403。
- **从中国 push GitHub 要走代理**：直连大概率 `TLS connection non-properly terminated`。用密封代理：
  `git -c http.proxy=http://127.0.0.1:7891 push ...`
- **import 解析**：务必 `pip install -e .` 到 webgym 环境，否则 `import webgym` 可能命中别处旧包。
- **磁盘**：截图很占空间，`save_path` 放大盘。`.gitignore` 已排除 `*.pt/*.png/*.log`，轨迹不会误传仓库。
- **成本**：gpt-5.4(视觉+推理) 调用量大，5k ≈ 几百元量级；预算紧时优先用更便宜的 new-api、关评判、或降步数。
- **OmniBoxes lease 泄漏**：kill -9 会跳过 lease 释放 → 容量缩水最终 503。优雅 kill；乱了就杀掉所有 omnibox 进程 + chromium 重启。

---

## 6. 输出格式（你收集到的东西）

`{save_path}/{train,test}_trajectories/*.pt`（torch 存的 `{'trajectories':[...], 'metadata':{...}}`）。
每条轨迹是 step 列表，每个 step 的 `response` 对象含：
- `raw_prompt`：**OpenAI 格式 message 数组**（JSON 字符串，图片是 `file://` 路径，多轮滑窗）——你要的训练数据主体。
- `raw_response`：gpt-5.4 原始输出。
- `answering_tokens`：解析出的 `{thought, action, tool_call, memory, ...}`。
截图在 `{save_path}/images/<task>/step_*.png`。
```

---

## UPDATE — 最终 harness(多站路由 + 有界 context)

这一版相对上面的单站版做了重大升级。要点:

### 运行环境变量(启动 rollout 前 export)
```
export POLICY_API_KEY_YUNWU=<yunwu key>            # 兜底站
export POLICY_API_KEY_NEWAPI=<new-api 账号1 key>    # 主力站(同一网关,RPM 独立)
export POLICY_API_KEY_NEWAPI2=<new-api 账号2 key>
export POLICY_API_KEY_NEWAPI3=<new-api 账号3 key>
export POLICY_NEWAPI_BASE_URL=https://<your-newapi-host>/v1/chat/completions  # 私有中转站地址,不入库
export CPU_CLUSTER_TOKEN=default_key
```
yunwu 的 base_url 是公开的 `https://yunwu.ai/v1`(写在配置里);new-api 的地址走 `POLICY_NEWAPI_BASE_URL`,所以仓库里不含私有域名。

### 多站策略路由(webgym/models/web_agent.py)
- `policy_config.api.endpoints` 列表:每个站 = base_url(或 base_url_env_var)+ api_key_env_var + weight + 可选 `fallback: true`。
- **分层 + 加固故障转移**:单次调用先在 PRIMARY(3 个 new-api,加权轮询)里逐站试;任一抽风(429/5xx/网络/空-200)→ 冷却该站、立刻试下一站;**一轮把所有 primary 试完仍失败 → 必定试 FALLBACK(yunwu)再说**;全部失败才放弃(且全站长冷却=配额耗尽时快速失败,不空转)。
- 401/403:账号级 → 冷却 120s;**内容审计类 403 → 只失败该次调用,不冷却整站**。
- yunwu 作为 `fallback: true`:**仅当 3 个 new-api 全不可用时才用**,平时 0 流量(省钱)。

### 有界 context(关键:防晚期步 payload 撑爆网关)
- 图片:滑窗最近 4 轮(≤5 张),已有界。
- **长期记忆 = gpt-5.4-mini 维护的滚动 `running_log`**(每步增量折叠,硬上限 ~150 词),注入每步输入第一条 user 作 "Notes so far";`webgym/models/web_agent.py:update_running_log`(失败则保留旧 log,不崩),折叠点在 `async_webgym` 步循环里,存进轨迹。
- 策略**输出不再含会膨胀的 Memory 字段**(只 Thought/Action/tool_call);旧的逐步全量文字摘要已废弃。
- 训练形态:**逐步 windowed 样本**(每步的 windowed 输入 → action),训练=推理一致。

### 任务过滤
`scripts/make_remaining_plan.py` 之外,bulk 跑用 `tasks_filtered.jsonl`(全池减去 ~138 个环境不可能任务:下载/打开本地文件、存本地文件、完成支付、带凭据登录)。

### 并发与稳定性教训
- 瓶颈是 **CPU 核数(浏览器渲染)**,不是内存;loadavg 高多为 I/O 等待,看真实 `%Cpu` idle。
- `server_size` 要 < omnibox 实例数(留余量),否则实例分配 churn → 503。64 核机器实测 **server_size 50 / omnibox 64** 稳定;128 会把 omnibox 后端搞垮(503 风暴)。
- **停 rollout 必须用 `scripts/stop_rollout.sh`**(连 worker 池一起杀),否则孤儿 worker 累积压垮 CPU。
- 清 omnibox 要杀 `deploy.py`(它有进程恢复会 respawn 浏览器)。
- 监控面板端口 **6006**(AutoDL 自定义服务直接代理)。
