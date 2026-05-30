# 遗留事项

本文档统一归档项目开发中的明确遗留事项。后续开发如果发现暂不处理但需要追踪的技术债、兼容性缺口或待验证场景，应按时间倒序追加到这里。

## 当前事项

### 2026-05-30：补充 `use_fp8_padding=True` 的 packed layout 对齐与测试

**问题**：TP v1 只支持 CP=1、非 FP8 padding 主路径。verl 在 `use_fp8_padding=True` 时会改变 packed padding 规则：先使用 `lcm(16, align_size)` 对每个序列长度做 padding，再对最后一行追加额外 padding 以满足 Transformer Engine 的总长度对齐要求。当前 `PackedBatchLayout` 尚未实现这套 FP8 padding 规则。

**方案**：后续启用 FP8 前，扩展 `PackedBatchLayout` builder，使其接收 `use_fp8_padding` 或等价配置，并严格对齐 `dependency/verl_v070/verl/models/mcore/util.py::preprocess_packed_seqs()` 的 FP8 padding 逻辑。同时补充 TP=2/4/8 下 `use_fp8_padding=True` 的 layout、restore index 和 backend 测试。
