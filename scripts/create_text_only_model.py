#!/usr/bin/env python3
"""Create a text-only Qwen3 model directory with proper config.json and weight remapping."""
import os
import json
import shutil

# Paths
multimodal_dir = os.path.expanduser("~/rollout-prefix/models/Qwen3.6-27B")
text_only_dir = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")

# Read multimodal config
with open(os.path.join(multimodal_dir, "config.json")) as f:
    full_config = json.load(f)

tc = full_config["text_config"]

# Create text-only Qwen3 config (using qwen3 model_type which transformers recognizes)
text_config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "attention_bias": tc.get("attention_bias", False),
    "attention_dropout": tc.get("attention_dropout", 0.0),
    "attn_output_gate": tc.get("attn_output_gate", True),
    "bos_token_id": tc.get("bos_token_id", 1),
    "eos_token_id": tc.get("eos_token_id", 2),
    "full_attention_interval": tc.get("full_attention_interval", 4),
    "head_dim": tc.get("head_dim", 256),
    "hidden_act": tc.get("hidden_act", "silu"),
    "hidden_size": tc.get("hidden_size", 5120),
    "initializer_range": tc.get("initializer_range", 0.02),
    "intermediate_size": tc.get("intermediate_size", 17408),
    "layer_types": tc.get("layer_types", []),
    "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
    "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
    "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
    "linear_num_value_heads": tc.get("linear_num_value_heads", 48),
    "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
    "max_position_embeddings": tc.get("max_position_embeddings", 262144),
    "num_attention_heads": tc.get("num_attention_heads", 24),
    "num_hidden_layers": tc.get("num_hidden_layers", 64),
    "num_key_value_heads": tc.get("num_key_value_heads", 4),
    "output_gate_type": tc.get("output_gate_type", "swish"),
    "partial_rotary_factor": tc.get("partial_rotary_factor", 0.25),
    "rms_norm_eps": tc.get("rms_norm_eps", 1e-06),
    "rope_theta": tc.get("rope_parameters", {}).get("rope_theta", 10000000),
    "rope_type": "default",
    "tie_word_embeddings": tc.get("tie_word_embeddings", False),
    "use_cache": True,
    "vocab_size": tc.get("vocab_size", 248320),
}

# Create text-only directory
os.makedirs(text_only_dir, exist_ok=True)

# Save config
with open(os.path.join(text_only_dir, "config.json"), "w") as f:
    json.dump(text_config, f, indent=2)
print("Created config.json in", text_only_dir)

# Copy tokenizer files
tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model"]
for fname in tokenizer_files:
    src = os.path.join(multimodal_dir, fname)
    dst = os.path.join(text_only_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

# Create text-only weight index by removing vision weights
with open(os.path.join(multimodal_dir, "model.safetensors.index.json")) as f:
    weight_index = json.load(f)

# Filter: keep only language_model and lm_head weights, remap prefixes
text_weight_map = {}
text_metadata = weight_index.get("metadata", {})

for key, shard_info in weight_index["weight_map"].items():
    # Skip vision weights
    if key.startswith("model.visual.") or key.startswith("model.") and not key.startswith("model.language_model."):
        if not key.startswith("model.language_model."):
            continue

    # Remap: model.language_model.X -> model.X
    new_key = key.replace("model.language_model.", "model.")
    text_weight_map[new_key] = shard_info

# Also keep lm_head weights
for key, shard_info in weight_index["weight_map"].items():
    if key == "lm_head.weight":
        text_weight_map[key] = shard_info

# Total size estimation
text_index = {
    "metadata": text_metadata,
    "weight_map": text_weight_map,
}

with open(os.path.join(text_only_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(text_index, f, indent=2)
print(f"Created weight index: {len(text_weight_map)} tensors (from {len(weight_index['weight_map'])} original)")

# Symlink the safetensors shard files (they contain both vision and language weights,
# but the index will only reference the language ones)
for fname in os.listdir(multimodal_dir):
    if fname.startswith("model.safetensors-") and fname.endswith(".safetensors"):
        src = os.path.join(multimodal_dir, fname)
        dst = os.path.join(text_only_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"  Symlinked {fname}")

print("Done! Text-only model at:", text_only_dir)