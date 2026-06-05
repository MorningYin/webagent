---
name: sft-purpose-vs-rl-reward
description: "WebGym 轨迹级成功率不是数据质量门槛——SFT 只教做题方式,正确性靠 RL + 段级 critic"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ef814bf-ea0d-4210-b65d-decb9ca2ed1a
---

WebGym 自带 reward(轨迹级 / 最终结果)的成功率很低(实测保留的 607 条真实轨迹里 reward==1 只有 10 条,1.6%,且经核实评判器没冤枉——抓的是 agent 弃疗 / 编造日期 / 答案不全等真问题),但**这对数据收集不致命**。

**Why:** 这批 rollout 数据的用途是 SFT,而 SFT 的目标只是让模型**学会怎么做题**——动作格式(`<tool_call>` computer_use)、分段式问题处理的推理结构。正确性不靠 SFT,而靠后续 **RL + 他们自己的段级 critic 细粒度 reward**(比 webgym 的轨迹级 reward 丰富得多;长轨迹信用分配问题用 segment 归因解决)。所以 webgym 轨迹级成功率**只是个最终结果标签,不是质量门槛**。

**How to apply:** 不要为了拉高 webgym 成功率去放宽评判器或挑简单任务;负样本(98%)是主体且有用。评判器严格是对的,保持。只在用户想要"轨迹级正样本作 DPO 锚点"时才关心 reward==1 数量(目前偏稀)。相关:[[webgym-closed-api-rollout-setup]]
