# 遗留事项

本文档统一归档项目开发中的明确遗留事项。后续开发如果发现暂不处理但需要追踪的技术债、兼容性缺口或待验证场景，应按时间倒序追加到这里。

## 当前事项

### 2026-06-04：prefix-sharing 可观测性日志分级与耗时拆分

**问题**：当前 profiler 分支先补充 expected/actual 复用统计，用于定位“理论应复用”和“运行时真实复用”是否一致；但还没有完整日志分级，也没有拆分 `plan_ms`、`build_kv_ms`、`attention_ms`、`restore_ms` 等耗时指标。若 expected/actual 已匹配但性能仍未提升，需要进一步定位慢点是复用比例不足、backend/kernel 路径低效，还是 Python `split/cat/store/load` 开销抵消收益。

**方案**：后续增加类似 `PREFIX_SHARING_OBSERVE=off|summary|layer|profile|debug` 的观测级别控制，并在 profile 级别补充阶段耗时。summary/layer 级别用于常驻问题定位，profile/debug 级别只在专项排查时打开，避免训练日志过量。

### 2026-05-30：补充 `use_fp8_padding=True` 的 packed layout 对齐与测试

**问题**：TP v1 只支持 CP=1、非 FP8 padding 主路径。verl 在 `use_fp8_padding=True` 时会改变 packed padding 规则：先使用 `lcm(16, align_size)` 对每个序列长度做 padding，再对最后一行追加额外 padding 以满足 Transformer Engine 的总长度对齐要求。当前 `PackedBatchLayout` 尚未实现这套 FP8 padding 规则。

**方案**：后续启用 FP8 前，扩展 `PackedBatchLayout` builder，使其接收 `use_fp8_padding` 或等价配置，并严格对齐 `dependency/verl_v070/verl/models/mcore/util.py::preprocess_packed_seqs()` 的 FP8 padding 逻辑。同时补充 TP=2/4/8 下 `use_fp8_padding=True` 的 layout、restore index 和 backend 测试。
