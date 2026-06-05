- [WebGym closed-API rollout setup](webgym-closed-api-rollout-setup.md) — gpt-5.4 policy via local LiteLLM proxy (:4000) over relay gateways (dkyx active, heiduoke disabled) + sealed 5-IP proxy chain; isolated conda env; towr-webgym is dead
- [SFT purpose vs RL reward](sft-purpose-vs-rl-reward.md) — webgym 轨迹级成功率(~1.6%)不是质量门槛;SFT 只教做题方式,正确性靠 RL + 段级 critic
- [Segment SFT pipeline](segment-sft-pipeline.md) — 段级信用分配 SFT 数据的 4 阶段离线流水线总览(收集→分段→段内摘要→构造);各阶段脚本与模型/网关绑定
- [clean_dataset artifacts](clean-dataset-artifacts.md) — /root/autodl-tmp 产物清单/schema/对齐键(2706 traj, 66285 步, 16386 段, 11297 个 L≥2);canonical 是 messages_with_segments.jsonl
- [Segmenter design](segmenter-design.md) — segmenter.py:gpt-5.5/dkyx 流式切段,四元组(observation/purpose/summary/result),按状态转移切而非源/路径,c8 稳
- [Within-segment summary design](within-segment-summary-design.md) — build_seg_summaries.py:双版本逐步 delta 笔记(structured+mini),前缀拼接+k 窗口,surface=same 也抓事实,长段分块
- [Message array & token budget](message-array-and-token-budget.md) — build_master_with_segments.py:窗口化 prompt 非全历史须重建母本,内联 beginner/stopper;Qwen3-VL 960 tok/图,window=2≈321M
- [Data migration & git state](data-migration-and-git-state.md) — 9.9G tar 包 server→local→target rsync;分支 segment-sft-pipeline 推到 webagent(非 origin/microsoft);密钥进 .env

### Insights / ideas（核心思想层）
- [Insight: segment credit assignment](insight-segment-credit-assignment.md) — 核心论断:信用分配的单位是 skill segment,不是 action(无局部信号)也不是 trajectory(太粗);人学技能不学单个动作
- [Insight: three-tier bounded memory](insight-three-tier-bounded-memory.md) — A 历史/B 段内/滑窗三层有界记忆;图像 token 随段长而非轨迹长;三层与 beginner/stopper 是同一系统(B 收口=stopper.summary,A 每条=过去的 stopper)
- [Insight: policy-intrinsic segmentation](insight-policy-intrinsic-segmentation.md) — 分段从外部标注变成策略自调用的两个工具;SFT 教格式(边界=assistant tool_call 而非 user XML),RL+critic 涌现技能;beginner 禁后见之明保证 train==inference
- [Insight: weak model as data tool](insight-weak-model-as-data-tool.md) — mini 修得了格式(unparseable 100%→17%)修不了正确性(reward==1 21% vs 54%);强模型管正确性,弱模型只做廉价结构化重写;别让弱模型做抽取
- [Insight: diversified data via randomization](insight-diversified-data-via-randomization.md) — 随机 k 窗口(图/文比例)+ 双 rolling 表示;构造期产出可随机化的原始对齐材料,样本实现延到下游,一份数据派生多变体
