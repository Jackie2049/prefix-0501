"""Tests for model_spec module."""

from __future__ import annotations

from prefix_sharing.core.model_spec import (
    AttentionLayerType,
    ModelSpec,
    QWEN3_6_27B,
)


class TestModelSpec:
    def test_all_full_attention_when_interval_zero(self):
        spec = ModelSpec(num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=64)
        assert spec.full_attention_interval == 0
        for i in range(8):
            assert spec.layer_type(i) == AttentionLayerType.FULL_ATTENTION

    def test_hybrid_attention_pattern(self):
        spec = ModelSpec(num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=64,
                         full_attention_interval=4)
        # Layers 0, 4 are full attention
        assert spec.layer_type(0) == AttentionLayerType.FULL_ATTENTION
        assert spec.layer_type(4) == AttentionLayerType.FULL_ATTENTION
        # Layers 1, 2, 3, 5, 6, 7 are linear attention
        for i in [1, 2, 3, 5, 6, 7]:
            assert spec.layer_type(i) == AttentionLayerType.LINEAR_ATTENTION

    def test_full_attention_layer_ids(self):
        spec = ModelSpec(num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=64,
                         full_attention_interval=4)
        assert spec.full_attention_layer_ids() == [0, 4]
        assert spec.num_full_attention_layers == 2

    def test_linear_attention_layer_ids(self):
        spec = ModelSpec(num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=64,
                         full_attention_interval=4)
        assert spec.linear_attention_layer_ids() == [1, 2, 3, 5, 6, 7]
        assert spec.num_linear_attention_layers == 6

    def test_gqa_group_size(self):
        spec = ModelSpec(num_hidden_layers=8, num_attention_heads=24,
                         num_key_value_heads=4, head_dim=256)
        assert spec.gqa_group_size == 6

    def test_from_hf_config_dict(self):
        hf_config = {
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 131072,
        }
        spec = ModelSpec.from_hf_config(hf_config)
        assert spec.num_hidden_layers == 64
        assert spec.full_attention_interval == 4
        assert spec.attn_output_gate is True
        assert spec.partial_rotary_factor == 0.25

    def test_from_hf_config_inferred_head_dim(self):
        hf_config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        spec = ModelSpec.from_hf_config(hf_config)
        assert spec.head_dim == 128  # 4096 / 32
        assert spec.num_key_value_heads == 32  # defaults to num_attention_heads


class TestQwen36Preset:
    def test_preset_values(self):
        assert QWEN3_6_27B.num_hidden_layers == 64
        assert QWEN3_6_27B.num_attention_heads == 24
        assert QWEN3_6_27B.num_key_value_heads == 4
        assert QWEN3_6_27B.head_dim == 256
        assert QWEN3_6_27B.full_attention_interval == 4
        assert QWEN3_6_27B.attn_output_gate is True
        assert QWEN3_6_27B.partial_rotary_factor == 0.25

    def test_qwen36_layer_distribution(self):
        # 16 full attention layers at indices 0,4,8,...,60
        full_ids = QWEN3_6_27B.full_attention_layer_ids()
        assert len(full_ids) == 16
        assert full_ids == [i * 4 for i in range(16)]

        # 48 linear attention layers
        linear_ids = QWEN3_6_27B.linear_attention_layer_ids()
        assert len(linear_ids) == 48
        assert len(full_ids) + len(linear_ids) == 64

    def test_qwen36_gqa(self):
        assert QWEN3_6_27B.gqa_group_size == 6  # 24 // 4
