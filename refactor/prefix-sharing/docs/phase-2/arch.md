# prefix-0501 架构与类图

---

## 1. 文件组织

`prefix-0501` 以整个 monorepo 为版本管理单元，包含调研、依赖的快照版本、正式模块开发三部分：

```text
prefix-0501/
├── survey/                      # 调研与 PoC
│   ├── flash-preference/        # DPO / preference 训练中的 prefix 复用参考
│   └── dpo-prefix-sharing/      # prompt/response 边界明确的 prefix sharing 参考
├── dependency/                  # 集成目标与参考实现快照
│   ├── verl_v070/               # RL 训练框架（actor logprob / update 主路径）
│   ├── megatron_v0150/          # Megatron-Core attention / packed THD / RoPE
│   └── PrefixTrain_dev/         # 在线 Trie prefix 复用 PoC（设计参考）
└── refactor/prefix-sharing/     # 正式开发仓库（本包）
    ├── prefix_sharing/          # 核心 Python 包
    │   ├── core/                # 框架无关核心语义
    │   ├── backends/            # Attention 后端适配（torch/cuda/cann）
    │   └── integrations/          # verl/Megatron 集成与 patch 管理
    ├── tests/                   # unit / integrated / system 测试
    └── docs/                    # 设计文档与 roadmap
```

### 1.1 各目录职责

| 目录 | 角色 | 与 prefix-sharing 的关系 |
|------|------|--------------------------|
| `refactor/prefix-sharing` | 正式实现 | 框架无关 core + backend 适配 + verl/Megatron 集成 |
| `dependency/verl_v070` | 集成宿主 | actor micro-batch、DataProto、vocab-parallel logprob |
| `dependency/megatron_v0150` | 集成宿主 | SelfAttention、packed THD、RoPE、TP/PP/CP 并行状态 |
| `dependency/PrefixTrain_dev` | 设计参考 | 在线 Trie、KV injection 思路；**不可复现** `clone().detach()` bug |
| `survey/*` | 场景参考 | 收益形态、测试场景；不直接继承实现 |

### 1.2 依赖关系

```mermaid
flowchart TB
    subgraph prefix0501["prefix-0501"]
        PS["refactor/prefix-sharing<br/>prefix_sharing 包"]
        VERL["dependency/verl_v070"]
        MEG["dependency/megatron_v0150"]
        PT["dependency/PrefixTrain_dev<br/>（设计参考）"]
        SURVEY["survey/*<br/>（场景参考）"]
    end

    PS -->|"monkey-patch / helper 调用"| VERL
    PS -->|"SelfAttention hook / RoPE"| MEG
    PT -.->|"Trie / KV injection 思路"| PS
    SURVEY -.->|"DPO / RL 场景"| PS

    VERL --> MEG
```

---

## 2. prefix-sharing 宏观架构

### 2.1 分层设计

```text
core/          核心特性层，实现框架无关的ps核心语义：检测 → 规划 → 元数据 → KV 存储 → logprob 恢复
integrations/  集成层，将ps特性集成到verl+megatron中。尽量做薄，控制对verl和megatron的侵入式修改，方便未来模块化插入
backends/      后端层，为上层屏蔽硬件差异。消费上层传入的统一 metadata，负责 RoPE / attention的处理等
```

### 2.2 宏观架构

三层架构与外部框架集成关系：

```plantuml
@startuml
skinparam componentStyle rectangle

title prefix-sharing 三层架构与外部集成

package "core/ 核心层" as CoreLayer {
    [TriePrefixDetector\n前缀检测] as Detector
    [PrefixSharingPlanner\n执行规划] as Planner
    [PrefixSharingBatchMeta\n元数据] as Meta
    [PrefixKVStore\nKV缓存] as KVStore
    [logprob restore\n精度恢复] as Restore
}

package "backends/ 后端层" as BackendLayer {
    [PrefixAttentionBackend\n抽象协议] as BackendProtocol
    [TorchReferenceBackend\nPyTorch实现] as TorchBackend
}

package "integrations/ 集成层" as IntegrationLayer {
    [VerlMCoreIntegration\nverl集成入口] as VerlIntegration
    [MegatronAttentionIntegration\nMegatron补丁] as MegatronIntegration
    [PrefixSharingRuntimeContext\n运行时上下文] as RuntimeContext
    [PatchManager\n补丁管理] as PatchMgr
}

package "外部框架" as External {
    [verl\nRL训练框架] as Verl
    [Megatron-Core\n分布式训练] as Megatron
}

' 【1】初始化阶段：安装补丁（monkey-patch）
VerlIntegration ..> Verl : 【1a】<color:red>monkey-patch
MegatronIntegration ..> Megatron : 【1b】<color:red>monkey-patch
MegatronIntegration --> PatchMgr : 【1c】安装补丁
VerlIntegration --> PatchMgr : 【1d】安装补丁

' 【2】计划阶段：检测与规划
VerlIntegration --> Planner : 【2a】调用规划
Planner ..> Detector : 【2b】调用检测
Detector --> Planner : 【2c】返回检测结果
Planner --> Meta : 【2d】生成计划
Planner ..> BackendProtocol : 【2e】<color:gray>消费metadata

' 【3】运行时阶段：执行与缓存
Verl --> RuntimeContext : 【3a】输入\n(input_ids/position_ids)
VerlIntegration --> RuntimeContext : 【3b】创建上下文
RuntimeContext --> Meta : 【3c】读取计划
RuntimeContext --> KVStore : 【3d】生命周期管理
RuntimeContext --> TorchBackend : 【3e】驱动执行\n(RoPE/KV构建/Attention)
TorchBackend --> KVStore : 【3f】prefix KV\n读/写
TorchBackend --> Verl : 【3g】输出\n(attention结果)

' 【4】恢复阶段：精度恢复
Meta --> Restore : 【4】提供restore位置

' 后端实现（架构约束）
BackendProtocol <|.. TorchBackend : <color:gray>实现协议

' 步骤说明
note right of External
  <b>执行步骤说明</b>
  ----
  <b>【1】初始化阶段</b>
  【1a】VerlMCoreIntegration 向 verl 安装 monkey-patch
  【1b】MegatronAttentionIntegration 向 Megatron 安装 monkey-patch
  【1c】Megatron 集成初始化 PatchManager
  【1d】Verl 集成初始化 PatchManager
  ----
  <b>【2】计划阶段</b>
  【2a】Verl 集成调用 Planner 开始规划
  【2b】Planner 调用 TriePrefixDetector 执行检测
  【2c】Detector 返回检测结果给 Planner
  【2d】Planner 生成 PrefixSharingBatchMeta 计划
  【2e】Planner 消费 Backend 协议校验 metadata
  ----
  <b>【3】运行时阶段</b>
  【3a】verl 向 RuntimeContext 提供输入数据
  【3b】Verl 集成创建 RuntimeContext 运行时上下文
  【3c】RuntimeContext 从 Meta 读取执行计划
  【3d】RuntimeContext 管理 KVStore 生命周期
  【3e】RuntimeContext 驱动 TorchBackend 执行
  【3f】TorchBackend 从 KVStore 读/写 prefix KV
  【3g】TorchBackend 返回 attention 结果给 verl
  ----
  <b>【4】恢复阶段</b>
  【4】Meta 提供 restore 位置给 Restore 模块恢复 logprob
end note

@enduml
```

### 2.3 核心流程

```mermaid
flowchart LR
    subgraph Input["输入"]
        IDS[input_ids]
    end

    subgraph Core["core/ 核心层"]
        DET["TriePrefixDetector<br/>检测共享前缀"]
        PLN["PrefixSharingPlanner<br/>生成执行计划"
        META["PrefixSharingBatchMeta<br/>元数据"]
        TRIM["batch_trim<br/>裁剪 batch"]
    end

    subgraph Backend["backends/ 后端层"]
        ROPE["apply_rope<br/>位置编码"]
        KV["build_kv<br/>KV Injection"]
        ATTN["attention<br/>注意力计算"]
    end

    subgraph Integration["integrations/ 集成层"]
        CTX["PrefixSharingRuntimeContext<br/>运行时上下文"]
        STORE["PrefixKVStore<br/>KV 缓存"]
        RESTORE["logprob restore<br/>Prefix-Last Restore"]
    end

    IDS --> DET
    DET --> PLN
    PLN --> META
    META --> TRIM
    META --> CTX
    CTX --> STORE
    TRIM --> ROPE
    ROPE --> KV
    KV --> STORE
    KV --> ATTN
    ATTN --> RESTORE
```

**流程说明**：

1. **检测阶段**：`TriePrefixDetector` 在 micro-batch 内识别共享前缀，建立 provider-reuser 关系
2. **规划阶段**：`PrefixSharingPlanner` 生成 metadata，决定每行的 Q 裁剪范围、KV 扩展方式、restore 位置
3. **裁剪阶段**：`batch_trim` 按 metadata 裁剪 input/label/mask，生成 packed THD 格式
4. **运行时阶段**：`PrefixSharingRuntimeContext` 维护 KVStore，每层 attention 通过 backend 执行 RoPE、KV Injection、attention
5. **恢复阶段**：`logprob restore` 从 provider 恢复 reuser 首 token 的 logprob，保证训练语义一致

### 2.3 精度方案

**One-Forward + KV Injection + Prefix-Last Restore**

为确保 prefix sharing 与普通 causal LM 训练的精度完全一致，采用三阶段协作方案：

| 组件 | 作用 | 关键设计 |
|------|------|----------|
| **One-Forward** | provider 完整计算，reuser 仅计算 suffix | provider 存储 prefix KV，reuser 复用之 |
| **KV Injection** | 让 reuser 的 suffix query 能 attend 到 provider 的 prefix KV | 在 attention 层将 provider prefix K/V 拼接到 reuser suffix K/V 之前 |
| **Prefix-Last Restore** | 恢复 reuser 首个 suffix token 的 logprob | 从 provider 的 prefix-last 位置读取 logits，计算首 token logprob 并插入 |

**关键约束**：缓存的 prefix KV **绝不 detach()**，保留在 autograd 计算图中确保梯度正确回流到 shared prefix。

---

## 3. Core 层类图

> **职责**：实现框架无关的 prefix sharing 核心语义——检测共享前缀、规划执行布局、管理 KV 生命周期、恢复 logprob 精度。

Core 层不依赖 PyTorch / verl / Megatron（`logprob.py` 与 `group_partition.py` 对 torch 为 lazy import 或可选）。

```plantuml
@startuml
skinparam classAttributeIconSize 0

class PrefixSharingConfigError <<exception>>

class "PrefixSharingConfig\n用户配置入口：检测阈值、后端选择、Phase约束校验" as PrefixSharingConfig {
    +enabled: bool
    +detector: str
    +backend: str
    +min_prefix_len: int
    +min_group_size: int
    +boundary_strategy: str
    +validate(model_config, integrate_mode)
}

class "PrefixDetector\n抽象基类，定义检测器接口" as PrefixDetector <<abstract>> {
    +detect(input_ids): PrefixDetectionResult
}

class "TriePrefixDetector\n在线Trie检测batch内per-sample复用关系" as TriePrefixDetector {
    +min_prefix_len: int
    +min_group_size: int
    +detect(input_ids): PrefixDetectionResult
}

class "_TrieNode\n内部Trie节点" as _TrieNode <<internal>> {
    +children: dict
    +indices: list
    +provider_index: int
}

class "PrefixReuseSpec\n单条复用关系：reuser借用哪个provider的prefix" as PrefixReuseSpec {
    +reuse_idx_in_batch: int
    +provider_idx_in_batch: int
    +prefix_len: int
}

class "PrefixGroup_Detector\n检测阶段的group视图" as PrefixGroup_Detector <<prefix_detector.PrefixGroup>> {
    +group_id: int
    +member_indices: tuple
    +prefix_len: int
    +provider_index: int
}

class "PrefixDetectionResult\n检测结果容器：provider_index/prefix_lens等" as PrefixDetectionResult {
    +batch_size: int
    +reuse_specs: tuple
    +groups: tuple
    +group_ids: tuple
    +provider_index: tuple
    +prefix_lens: tuple
    +is_provider: tuple
}

class "PrefixSharingPlanner\n规划器：检测结果→执行计划(metadata)" as PrefixSharingPlanner {
    +config: PrefixSharingConfig
    +detector: TriePrefixDetector
    +plan(input_ids): PrefixSharingBatchMeta
    +plan_from_detection(input_ids, detection): PrefixSharingBatchMeta
}

class "PrefixLastRestoreSpec\n单条restore计划：从provider哪个位置恢复" as PrefixLastRestoreSpec {
    +reuse_idx_in_batch: int
    +provider_idx_in_batch: int
    +provider_prefix_last_pos: int
    +reuse_first_suffix_label_pos: int
    +output_slot: int
    +group_id: int
}

class "PrefixSharingBatchMeta\n单次micro-batch完整计划：裁剪/扩展/累积长度" as PrefixSharingBatchMeta {
    +forward_id: int
    +micro_batch_id: int
    +batch_size: int
    +reuse_specs: list
    +prefix_lens: list
    +kept_lengths_q: list
    +expanded_lengths_kv: list
    +cu_seqlens_q: list
    +cu_seqlens_kv: list
    +prefix_last_restore: list
    +has_sharing: bool
    +is_reuser(idx): bool
    +q_range_for_batch(idx): Range
    +kv_range_for_batch(idx): Range
}

class "TrimmedBatch\n裁剪后的batch视图：保留片段+packed格式" as TrimmedBatch {
    +rows: list
    +flattened: list
    +cu_seqlens: list
}

class "PrefixKVSlotId\nKV槽位标识：forward/micro_batch/layer/sample/tp_rank" as PrefixKVSlotId {
    +forward_id: int
    +micro_batch_id: int
    +layer_id: int
    +sample_idx_in_batch: int
    +tp_rank: int
}

class "StoredPrefixKV\n存储的KV条目" as StoredPrefixKV {
    +key_tensor: Any
    +value_tensor: Any
    +prefix_len: int
}

class "PrefixKVStore\nKV缓存管理：单次forward生命周期，不detach" as PrefixKVStore {
    -entries: dict
    -closed: bool
    +store(slot_id, key, value, prefix_len)
    +load(slot_id): StoredPrefixKV
    +contains(slot_id): bool
    +clear()
    +close()
}

class "PrefixGroup_Partition\nPartition阶段的group视图" as PrefixGroup_Partition <<group_partition.PrefixGroup>> {
    +group_id: str
    +sample_indices: tuple
    +original_tokens: int
    +estimated_compute_tokens: int
    +reusable_prefix_tokens: int
}

class "PrefixGroupPartition\nDP负载均衡结果：prefix group感知的rank分配" as PrefixGroupPartition {
    +dp_rank_to_indices: tuple
    +dp_rank_to_group_ids: tuple
    +group_workloads: dict
    +fallback_reason: str
    +is_fallback: bool
}

PrefixSharingConfig ..> PrefixSharingConfigError : raises
PrefixDetector <|-- TriePrefixDetector
TriePrefixDetector --> _TrieNode : uses
TriePrefixDetector ..> PrefixDetectionResult : produces
PrefixDetectionResult *-- PrefixReuseSpec
PrefixDetectionResult *-- PrefixGroup_Detector

PrefixSharingPlanner --> PrefixSharingConfig
PrefixSharingPlanner --> TriePrefixDetector
PrefixSharingPlanner ..> PrefixDetectionResult : consumes
PrefixSharingPlanner ..> PrefixSharingBatchMeta : produces
PrefixSharingBatchMeta *-- PrefixReuseSpec
PrefixSharingBatchMeta *-- PrefixLastRestoreSpec

TrimmedBatch ..> PrefixSharingBatchMeta : sliced by keep_ranges

PrefixKVStore *-- StoredPrefixKV
StoredPrefixKV --> PrefixKVSlotId : keyed by

PrefixGroupPartition *-- PrefixGroup_Partition

@enduml
```

---

## 4. Backends 层类图

> **职责**：消费统一 metadata，执行硬件相关的 attention 计算——RoPE 应用、KV Injection、核心 attention 算子，屏蔽 CPU/CUDA/CANN 硬件差异。

Backend 通过 `Protocol` 定义接口，reference 实现基于纯 PyTorch；CUDA/CANN 当前继承 `TorchReferenceBackend`（Phase 2.3 将接入 flash-attn / TE / CANN 融合算子）。

```plantuml
@startuml
skinparam classAttributeIconSize 0

class "BackendCapabilities\n能力声明：设备支持/特性支持矩阵" as BackendCapabilities {
    +name: str
    +supports_cpu: bool
    +supports_cuda: bool
    +supports_cann: bool
    +supports_different_q_kv_lengths: bool
    +supports_prefix_last_restore: bool
    +supports_fused_rope: bool
    +supports_context_parallel: bool
    +supports_pipeline_parallel: bool
}

class "PrefixAttentionBackend\n抽象协议：RoPE/KV构建/attention接口" as PrefixAttentionBackend <<Protocol>> {
    +capabilities: BackendCapabilities
    +validate(config, model_config)
    +apply_rope(query, key, meta): tuple
    +build_kv(key, value, store, meta, layer_id, tp_rank): tuple
    +attention(query, key, value, meta): Any
}

class "TorchReferenceBackend\n纯PyTorch参考实现：精度验证与fallback" as TorchReferenceBackend {
    +capabilities: BackendCapabilities
    +validate(config, model_config)
    +apply_rope(query, key, meta, rope_fn)
    +build_kv(key, value, store, meta, layer_id, tp_rank)
    +attention(query, key, value, meta)
}

class "CudaReferenceBackend\n高性能CUDA后端占位(Phase 2.3)" as CudaReferenceBackend <<Phase 2.3 placeholder>>

class "CannReferenceBackend\n华为NPU CANN后端占位(Phase 2.3)" as CannReferenceBackend <<Phase 2.3 placeholder>>

PrefixAttentionBackend <|.. TorchReferenceBackend : implements
TorchReferenceBackend --> BackendCapabilities
CudaReferenceBackend --|> TorchReferenceBackend
CannReferenceBackend --|> TorchReferenceBackend

TorchReferenceBackend ..> PrefixSharingConfig : validates
TorchReferenceBackend ..> PrefixSharingBatchMeta : consumes
TorchReferenceBackend ..> PrefixKVStore : read/write

@enduml
```

### 4.1 Backend 运行时 KV 扩展逻辑

```text
对 batch 中每个 sample i（layer L, tp_rank R）:

  provider (非 reuser):
    store[forward_id, micro_batch_id, L, i, R] = full K/V
    expanded_KV = full K/V

  reuser (provider_index[i] != i, prefix_lens[i] > 0):
    load provider slot → slice prefix K/V
    expanded_KV = concat(provider_prefix_KV, suffix_KV)
    store[forward_id, micro_batch_id, L, i, R] = expanded_KV  (transitive reuse)
```

---

## 5. Integrations 层类图

> **职责**：薄适配层——管理 monkey-patch 生命周期、传递运行时上下文、衔接 verl/Megatron 数据流，将 prefix sharing 能力注入训练框架。

Integrations 负责 patch 生命周期、runtime context 传递，以及 verl actor 微批预处理/后处理。

```plantuml
@startuml
skinparam classAttributeIconSize 0

class IntegrationUnavailable <<exception>>

class "PatchManager\nmonkey-patch管理：安装/回滚属性替换" as PatchManager {
    -records: list
    +patch_attr(target, attr_name, replacement)
    +handle(): PatchHandle
    +rollback()
}

class "PatchHandle\npatch句柄：启用/禁用补丁" as PatchHandle {
    -records: list
    -active: bool
    +active: bool
    +disable()
}

class "_PatchRecord\n内部patch记录" as _PatchRecord <<internal>> {
    +target: Any
    +attr_name: str
    +original: Any
    +replacement: Any
}

class "ParallelEnv\n并行环境快照：DP/TP/PP/CP/EP rank+world size" as ParallelEnv {
    +dp_rank: int
    +dp_world_size: int
    +tp_rank: int
    +tp_world_size: int
    +pp_rank: int
    +pp_world_size: int
    +cp_rank: int
    +cp_world_size: int
    +ep_rank: int
    +sequence_parallel: bool
    +trace_prefix(): str
}

class "PrefixSharingStats\n统计信息：trace/reuse count/saved tokens" as PrefixSharingStats {
    +trace_key: str
    +dp_rank: int
    +batch_size: int
    +reuse_count: int
    +saved_tokens_q: int
    +fallback_reason: str
}

class "PrefixSharingRuntimeContext\n运行时上下文：聚合meta/store/backend/restore" as PrefixSharingRuntimeContext {
    +meta: PrefixSharingBatchMeta
    +store: PrefixKVStore
    +backend: Any
    +kept_position_ids: Any
    +restore_positions: list
    +parallel_env: ParallelEnv
    +stats: PrefixSharingStats
}

class "MegatronAttentionIntegration\nMegatron集成入口：安装attention patch" as MegatronAttentionIntegration {
    +config: PrefixSharingConfig
    +backend: Any
    +install(model_config): PatchHandle
}

class "VerlMCoreBatchAdapter\nverl微批适配：准备batch/开context/恢复logprobs" as VerlMCoreBatchAdapter {
    +config: PrefixSharingConfig
    +planner: PrefixSharingPlanner
    +prepare_micro_batch(...): VerlMCorePreparedBatch
    +prepared_context(prepared): ContextManager
    +restore_logprobs(...): list
}

class "VerlMCorePreparedBatch\n准备好的batch：metadata+裁剪后的数据" as VerlMCorePreparedBatch {
    +meta: PrefixSharingBatchMeta
    +input_ids: TrimmedBatch
    +labels: TrimmedBatch
    +loss_masks: TrimmedBatch
}

class "VerlMCoreIntegration\nverl集成入口：聚合adapter和attention" as VerlMCoreIntegration {
    +config: PrefixSharingConfig
    +backend: Any
    +batch_adapter: VerlMCoreBatchAdapter
    +install(model_config): PatchHandle
}

class "MegatronActorPreparedMicroBatch\nMegatron actor专用batch：含dense position" as MegatronActorPreparedMicroBatch {
    +meta: PrefixSharingBatchMeta
    +backend: Any
    +kept_position_ids: Any
    +restore_positions: list
    +parallel_env: ParallelEnv
}

class "DensePrefixLastRestoreSpec\ndense格式restore：记录provider/reuse位置" as DensePrefixLastRestoreSpec {
    +reuse_idx_in_batch: int
    +provider_idx_in_batch: int
    +provider_dense_pos: int
    +reuse_dense_pos: int
}

PatchManager *-- _PatchRecord
PatchManager ..> PatchHandle : creates
PatchHandle --> _PatchRecord

PrefixSharingRuntimeContext --> PrefixSharingBatchMeta
PrefixSharingRuntimeContext --> PrefixKVStore
PrefixSharingRuntimeContext --> ParallelEnv
PrefixSharingRuntimeContext --> PrefixSharingStats

MegatronAttentionIntegration --> PrefixSharingConfig
MegatronAttentionIntegration --> PatchManager

VerlMCoreIntegration --> PrefixSharingConfig
VerlMCoreIntegration --> VerlMCoreBatchAdapter
VerlMCoreIntegration --> MegatronAttentionIntegration
VerlMCoreBatchAdapter --> PrefixSharingPlanner
VerlMCoreBatchAdapter ..> VerlMCorePreparedBatch : produces

MegatronActorPreparedMicroBatch --> PrefixSharingBatchMeta
MegatronActorPreparedMicroBatch --> DensePrefixLastRestoreSpec
MegatronActorPreparedMicroBatch --> ParallelEnv

@enduml
```

### 5.1 集成入口函数（模块级 API）

| 函数 | 模块 | 用途 |
|------|------|------|
| `enable_prefix_sharing()` | `verl_mcore` | 安装 Megatron attention patch |
| `prepare_megatron_actor_micro_batch()` | `verl_mcore` | verl actor 微批 in-place 裁剪 + 生成 prepared batch |
| `megatron_actor_prefix_sharing_context()` | `verl_mcore` | 打开 runtime context（供 patched attention 读取） |
| `restore_megatron_actor_log_probs()` | `verl_mcore` | forward 后恢复 reuser 首 suffix logprob |
| `maybe_run_prefix_sharing_attention()` | `megatron_runtime` | patched SelfAttention.forward 内的 attention 分支 |
| `reorder_dataproto_for_prefix_group_dp_balance()` | `verl_dp_balance` | Phase 2 DP 按 prefix group 重排 DataProto |
| `current_parallel_env()` | `parallel_env` | 读取 Megatron parallel_state 构建 ParallelEnv |

---

## 6. 端到端运行时序列

verl + Megatron actor 一次 forward 的主路径：

```mermaid
sequenceDiagram
    participant Verl as verl Actor
    participant Prep as prepare_megatron_actor_micro_batch
    participant Planner as PrefixSharingPlanner
    participant Ctx as prefix_sharing_context
    participant Attn as Megatron SelfAttention (patched)
    participant Runtime as maybe_run_prefix_sharing_attention
    participant Backend as TorchReferenceBackend
    participant Store as PrefixKVStore
    participant Restore as restore_megatron_actor_log_probs

    Verl->>Prep: micro-batch (input_ids, mask, position_ids)
    Prep->>Planner: plan(valid token sequences)
    Planner-->>Prep: PrefixSharingBatchMeta + trimmed batch
    Prep-->>Verl: (trimmed batch, MegatronActorPreparedMicroBatch)

    Verl->>Ctx: megatron_actor_prefix_sharing_context(prepared)
    Ctx->>Store: new PrefixKVStore()

    loop each Transformer layer
        Verl->>Attn: forward(Q, K, V, rotary_pos_emb, packed_seq_params)
        Attn->>Runtime: maybe_run_prefix_sharing_attention(...)
        Runtime->>Runtime: apply RoPE with kept_position_ids
        Runtime->>Backend: build_kv(K, V, store, meta, layer_id, tp_rank)
        Backend->>Store: store/load provider prefix KV
        Runtime->>Backend: attention(Q, expanded_K, expanded_V, meta)
        Backend-->>Attn: core_attn_out
    end

    Verl->>Restore: restore_megatron_actor_log_probs(logits, labels, log_probs)
    Restore->>Ctx: read restore_positions from context
    Restore-->>Verl: restored log_probs

    Ctx->>Store: close()
```

---

## 7. 测试分层

```text
tests/
├── unit_test/           # core 语义、config、planner、store、parallel_env
├── integrated_test/     # patch 生命周期、verl helper、DP balance
│   └── optional/        # 需 torch / verl / GPU 的 optional 测试
└── system_test/         # Phase 1 core 端到端系统测试
```

---

## 8. Phase 2 演进中的架构扩展点

当前类图已预留以下 Phase 2 扩展（详见 `parallel-plan.md`、`design-history.md`）：

| 扩展点 | 涉及类/模块 | Phase 2 目标 |
|--------|-------------|--------------|
| TP local K/V shard | `PrefixKVSlotId.tp_rank`、`ParallelEnv.tp_*` | 各 TP rank 独立 slot |
| DP 生命周期隔离 | `PrefixSharingBatchMeta.forward_id/micro_batch_id` | micro-batch / grad accumulation 不串 cache |
| PP stage-local store | `PrefixKVStore` + `ParallelEnv.pp_*` | 各 PP stage 独立 store |
| CP KV exchange | `ParallelEnv.cp_*` + backend | 跨 rank restore / exchange |
| BackendCapabilities 扩展 | `BackendCapabilities` | 声明 parallel / fused path 能力矩阵 |
| DP 负载均衡 | `group_partition` + `verl_dp_balance` | uid 驱动的 prefix group partition |

---

## 9. 命名冲突说明

包内存在两个同名但语义不同的 `PrefixGroup`：

| 类 | 模块 | 语义 |
|----|------|------|
| `PrefixGroup` | `core/prefix_detector.py` | 检测阶段兼容/debug 视图，按 `(provider_index, prefix_len)` 分组 |
| `PrefixGroup` | `core/group_partition.py` | DP 调度阶段 workload 估算单元，按 verl `uid` 等 group id 分组 |

文档与类图中通过后缀 `_Detector` / `_Partition` 区分；源码中尚未重命名，调用方需注意 import 来源。
