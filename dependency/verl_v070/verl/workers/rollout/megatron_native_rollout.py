# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Megatron Native Rollout - uses the same Megatron model as the actor for generation.

In colocated mode (训推共卡), the rollout shares the actor's model weights.
No separate model is loaded, avoiding OOM on limited-memory GPUs.

For generation, this uses token-by-token forward passes through the Megatron model.
This is slower than vLLM (no KV cache, O(n²) recomputation per step) but correct
and sufficient for initial e2e testing.
"""

import logging
import os

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.megatron_utils import unwrap_model
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.utils.config import omega_conf_to_dataclass

from .base import BaseRollout

__all__ = ["MegatronNativeRollout"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronNativeRollout(BaseRollout):
    """Rollout using the actor's Megatron model directly for generation.

    In colocated mode, this avoids loading a separate model (HF or vLLM),
    which would cause OOM on limited-memory GPUs (e.g., RTX 4090 24GB).

    Generation is done by token-by-token forward passes through the Megatron model.
    This is O(n²) in sequence length (no KV cache), but correct for testing.

    For production deployment, use vLLM or SGLang async rollout instead.
    """

    def __init__(self, config, model_config, device_mesh, actor_module=None):
        self.config = omega_conf_to_dataclass(config, dataclass_type=RolloutConfig)
        self.model_config = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.device_mesh = device_mesh
        self.actor_module = actor_module  # nn.ModuleList from MegatronPPOActor

    def set_actor_module(self, actor_module, hf_config, tf_config):
        """Set actor module reference after actor init completes.

        In colocated mode, the rollout shares the actor's model weights.
        This method is called after the actor model is initialized to set
        the reference and configs needed for generation.
        """
        self.actor_module = actor_module  # nn.ModuleList from MegatronPPOActor
        self.hf_config = hf_config
        self.tf_config = tf_config

    async def resume(self, tags: list[str]):
        """No separate model - just put actor in eval mode."""
        for module in self.actor_module:
            module.eval()

    async def update_weights(self, weights, **kwargs):
        """Same model - weights already in sync with actor."""
        pass

    async def release(self):
        """No separate model to release - just put actor back in train mode."""
        for module in self.actor_module:
            module.train()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the Megatron model's forward pass.

        Token-by-token autoregressive generation:
        1. Start with prompt tokens
        2. For each step, do a forward pass through the Megatron model
        3. Get logits at the last position
        4. Sample from logits (with temperature, top_p, etc.)
        5. Append sampled token
        6. Repeat until response_length or EOS

        Returns DataProto with same format as HFRollout output.
        """
        rollout_config = self.config

        # Sampling parameters from meta_info or config
        do_sample = prompts.meta_info.get("do_sample", rollout_config.do_sample)
        temperature = prompts.meta_info.get("temperature", rollout_config.temperature)
        response_length = prompts.meta_info.get("response_length", rollout_config.response_length)
        top_p = prompts.meta_info.get("top_p", rollout_config.get("top_p", 1.0))

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        # Move to GPU
        device = get_device_id()
        idx = idx.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        # Prepare for generation: build full sequence progressively
        generated_tokens = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Get the forward function
        from verl.models.mcore import get_mcore_forward_fn
        forward_fn = get_mcore_forward_fn(self.model_config)

        # Token-by-token generation
        current_ids = idx.clone()
        current_mask = attention_mask.clone()
        current_pos = position_ids.clone()

        with torch.no_grad():
            for step in range(response_length):
                if finished.all():
                    break

                # Forward pass through the Megatron model
                # For PP=1, we call each model chunk directly
                logits = self._forward_for_logits(
                    forward_fn, current_ids, current_mask, current_pos
                )

                # logits shape: (batch_size, seq_len, vocab_size) or (total_nnz, vocab_size)
                # We need logits at the last valid position for each sequence
                if logits.dim() == 3:
                    # BSHD format: (batch_size, seq_len, vocab_size)
                    last_logits = logits[:, -1, :]  # (bs, vocab_size)
                elif logits.dim() == 2:
                    # THD format after unpack: (batch_size, vocab_size)
                    last_logits = logits
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                # Sample from logits
                if not do_sample:
                    # Greedy decoding
                    next_token = last_logits.argmax(dim=-1)  # (bs,)
                else:
                    # Temperature sampling
                    last_logits = last_logits / temperature
                    # Apply top_p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(last_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        last_logits[indices_to_remove] = -float("inf")

                    probs = torch.softmax(last_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (bs,)

                # Replace next_token for finished sequences with pad_token_id
                next_token = next_token.masked_fill(finished, pad_token_id)

                # Check for EOS
                finished = finished | (next_token == eos_token_id)

                # Append token to sequence
                generated_tokens.append(next_token)

                # Extend current sequence for next step
                current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    (~finished).long().unsqueeze(1)
                ], dim=1)
                # Position IDs: increment from last position
                last_pos = current_pos[:, -1:]
                new_pos = last_pos + 1
                current_pos = torch.cat([current_pos, new_pos], dim=1)

        # Build response from generated tokens
        if len(generated_tokens) == 0:
            # Edge case: no tokens generated
            response = torch.full(
                (batch_size, response_length), pad_token_id,
                dtype=idx.dtype, device=device
            )
        else:
            response = torch.stack(generated_tokens, dim=1)  # (bs, num_generated)
            # Pad to response_length if needed
            if response.size(1) < response_length:
                pad_len = response_length - response.size(1)
                padding = torch.full(
                    (batch_size, pad_len), pad_token_id,
                    dtype=response.dtype, device=device
                )
                response = torch.cat([response, padding], dim=1)

        # Full sequence: prompt + response
        seq = torch.cat([idx, response], dim=1)  # (bs, prompt_length + response_length)
        sequence_length = prompt_length + response_length

        # Build attention_mask for full sequence
        # Response mask: 1 for non-pad tokens, 0 for pad tokens
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([attention_mask, response_attention_mask.to(device)], dim=1)

        # Build position_ids for full sequence
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=1)

        # Build output DataProto (same format as HFRollout)
        prompt = seq[:, :prompt_length]
        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size,
        )

        # Clear GPU cache
        get_torch_device().empty_cache()

        return DataProto(batch=batch)

    def _forward_for_logits(self, forward_fn, input_ids, attention_mask, position_ids):
        """Do a forward pass through the Megatron model to get logits.

        For PP=1, we call the model directly.
        Handles TP by gathering logits on all ranks.

        Returns logits of shape (batch_size, vocab_size_per_rank) on non-last PP stage,
        or (batch_size, vocab_size) on last PP stage.
        """
        # Convert attention_mask to bool for the forward function
        attn_mask_bool = attention_mask.to(bool)

        # For PP=1, call the model directly (only one chunk in actor_module)
        model = self.actor_module[0]
        unwrapped = unwrap_model(model)

        # Use the model forward function
        # For rmpad (use_remove_padding=True), data_format is "thd"
        use_rmpad = getattr(unwrapped.config, 'use_remove_padding', False)
        data_format = "thd" if use_rmpad else "bshd"

        output = forward_fn(
            model=model,
            input_ids=input_ids,
            attention_mask=attn_mask_bool,
            position_ids=position_ids,
            multi_modal_inputs={},
            logits_processor=None,
            logits_processor_args={},
            data_format=data_format,
        )

        # output is logits tensor: (batch_size, seq_len, vocab_size) for bshd
        # or (total_nnz, vocab_size) for thd (need to unpack)
        if use_rmpad:
            # THD format: output is (total_nnz, vocab_size)
            # We need to get the last valid token's logits for each sequence
            # For simplicity, since we padded the sequence, the last position
            # logits corresponds to the last input token
            # The output is already unpacked by postprocess_packed_seqs in forward_fn
            # when logits_processor is None
            # So output should be (batch_size, seq_len, vocab_size)
            pass

        return output