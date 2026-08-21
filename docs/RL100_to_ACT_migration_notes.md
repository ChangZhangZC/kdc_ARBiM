# RL-100 PPO / Uni-PPO / Buffer 阅读总结：ACT Offline PPO 迁移指南

## 1. 阅读范围

已精读：

-   `ppo.py`
-   `uni_ppo.py`
-   `buffer.py`

目标：

将 RL-100 的 Offline PPO Post-training 思想迁移到 ACT Action Chunk
Policy。

核心迁移目标：

    ACT IL checkpoint
            ↓
    构造 Offline RL 数据
            ↓
    训练 Critic / IQL
            ↓
    计算 Action Chunk Advantage
            ↓
    PPO update ACT policy

------------------------------------------------------------------------

# 一、整体迁移原则

## 1. PPO 核心保持不变

保留：

\[ r_t=`\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}`{=tex} \]

以及：

\[ L=-min(rA,clip(r,1-`\epsilon`{=tex},1+`\epsilon`{=tex})A) \]

需要改变的是：

-   action 如何定义
-   log probability 如何计算
-   advantage 如何计算
-   数据如何组织

------------------------------------------------------------------------

# 二、ppo.py 对 ACT 的迁移

## 保留部分

### PPO Actor Update

保留：

-   ratio calculation
-   PPO clipping
-   entropy regularization
-   optimizer update
-   gradient clipping

这些属于通用 PPO。

------------------------------------------------------------------------

## 需要修改部分

### 1. Action representation

原 PPO：

    a_t

ACT：

    a_chunk=[a_t,...,a_{t+K}]

需要重新定义：

\[ `\pi`{=tex}(a\_{chunk}\|s) \]

------------------------------------------------------------------------

### 2. log_prob

ACT 原始 deterministic policy 没有：

\[ log`\pi`{=tex}(a\|s) \]

需要改成 stochastic ACT：

例如：

\[ a`\sim `{=tex}N(`\mu`{=tex},`\sigma`{=tex}) \]

然后：

    log_prob = distribution.log_prob(action_chunk)

------------------------------------------------------------------------

# 三、uni_ppo.py 对 ACT 的迁移

## 最重要思想

RL-100 将 Diffusion Action Chunk 看成：

    one chunk = one PPO action

对应：

\[ r\_{chunk}A\_{chunk} \]

ACT 第一版建议沿用。

------------------------------------------------------------------------

# 推荐第一版 ACT PPO 设计

## Ratio

使用：

    scalar chunk ratio

即：

\[ log`\pi`{=tex}(a\_{chunk}) =
`\sum`{=tex}*{k,d}log`\pi`{=tex}(a*{k,d}) \]

得到：

一个 ratio。

------------------------------------------------------------------------

## Advantage

推荐：

    scalar_iql

即：

\[ A=Q(s,a\_{chunk})-V(s) \]

原因：

-   最接近 RL-100
-   不需要 dynamics
-   与 ACT chunk action 对齐

------------------------------------------------------------------------

# 四、uni_ppo.py 中需要替换的 DP 专属部分

## 可以删除 / 重写

### Diffusion timestep loop

例如：

    noise_scheduler.timesteps
    denoise step
    eta
    old_all_x
    old_all_next_x

这些都是 Diffusion Policy 专属。

ACT 不需要。

------------------------------------------------------------------------

### Diffusion log probability

替换：

    all_step_logprob()

ACT 应改为：

    old_dist.log_prob(action_chunk)
    new_dist.log_prob(action_chunk)

------------------------------------------------------------------------

# 五、buffer.py 对 ACT 数据转换

## RL-100 Offline 数据格式

要求：

    observations
    actions
    next_observations
    rewards
    terminals

------------------------------------------------------------------------

## ACT 推荐格式

不要使用单步 action：

\[ (s_t,a_t,s\_{t+1}) \]

而使用：

\[ (s_t,a\_{t:t+K},R,s\_{t+K},done) \]

即：

    state
    action_chunk
    chunk_reward
    next_state
    done

------------------------------------------------------------------------

# 六、IL → Offline RL 数据转换

推荐流程：

    LeRobot dataset
            ↓
    episode读取
            ↓
    构造 action chunk
            ↓
    reward annotation
            ↓
    next observation
            ↓
    terminal判断
            ↓
    Offline RL Dataset

------------------------------------------------------------------------

# 七、数据源选择建议

## 推荐：

使用已经清洗好的 LeRobot 数据。

原因：

已经包含：

-   observation同步
-   action同步
-   episode结构
-   数据清洗

Rosbag 只作为：

-   reward额外信号来源
-   力传感器
-   接触事件
-   外部成功检测

不要重新从 Rosbag 建整个 pipeline。

------------------------------------------------------------------------

# 八、三个文件函数总结

## ppo.py

### PPO 类

负责标准 PPO Actor-Critic 更新框架。

### update

执行一次 PPO policy/value 更新。

### compute_advantage

计算 TD error 和 GAE advantage。

### evaluate

评估 policy 性能。

### save/load

保存和恢复模型。

------------------------------------------------------------------------

## uni_ppo.py

### UniPPO 类

扩展 PPO 支持 Diffusion Policy / Action Chunk PPO。

### update_distribution

核心 Offline PPO Actor 更新函数。

负责：

-   old policy log probability
-   new policy log probability
-   ratio
-   PPO loss
-   backward

### \_get_offline_chunk_modes

选择 chunk ratio 和 advantage 模式。

### \_sum_logprob_event_dims

将 action dimension 的 log probability 求和得到 joint log probability。

### \_apply_chunk_adv_clip

限制 advantage 数值范围。

### save_critic/load_critic

保存加载 critic。

### transfer2online

Offline 到 Online RL 阶段切换。

### dp_align_update_no_share

Online PPO 更新。

### distill_update

Diffusion policy 蒸馏。

------------------------------------------------------------------------

## buffer.py

### OnlineReplayBuffer

管理 RL transition。

### store

存储：

\[ (s,a,r,s',done) \]

### compute_return

计算 discounted return。

### compute_advantage

计算 GAE advantage。

### sample

随机采样 RL batch。

### sample_all

导出 offline dataset。

### OfflineReplayBuffer

从已有 dataset 创建 RL buffer。

### load_dataset

直接加载 Offline RL 数据。

### load_filter_dataset

按照 return 过滤高质量数据。

### reward_normalize

reward scaling。

### normalize_state

state normalization。

------------------------------------------------------------------------

# 九、阅读优先级评价

## 必须深入

1.  uni_ppo.py

原因：

决定 ACT PPO 如何实现。

重点：

-   log_prob
-   ratio
-   chunk advantage

2.  critic.py（下一阶段）

原因：

决定：

\[ Q,V,A \]

如何获得。

3.  train_ddp.py

原因：

理解完整训练循环。

------------------------------------------------------------------------

## 了解即可

buffer.py

重点：

数据格式。

不需要完全照搬。

------------------------------------------------------------------------

## DP 专属，可跳过

-   diffusion denoise loop
-   eta
-   diffusion timestep
-   distillation

------------------------------------------------------------------------

# 十、下一阶段路线

    buffer.py
            ↓
    critic.py
            ↓
    train_ddp.py finetune_dp3()
            ↓
    ACT stochastic policy design
            ↓
    ACT Offline PPO prototype
