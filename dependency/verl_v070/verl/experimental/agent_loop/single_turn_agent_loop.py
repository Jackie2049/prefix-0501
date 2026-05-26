# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import hashlib
import logging
import os
import random
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        # 前缀配置：同一 prompt 的 n=8 条 response 共享相同前缀
        self.prefix_len: int = 64             # 前缀 token 数，按需调整
        self.vocab_range: int = 32000         # 模型 vocab size，确保 token id 有效

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        logger.warning("\n\n\nBegin to generate prefix sequences")
        # 3. 根据 prompt 生成确定性前缀（同 prompt 同前缀，不同 prompt 不同前缀）
        prompt_hash = hashlib.md5(repr(messages).encode()).hexdigest()
        seed = int(prompt_hash[:8], 16)
        rng = random.Random(seed)
        prefix_token_ids: list[int] = [
            rng.randint(1, self.vocab_range - 1) for _ in range(self.prefix_len)
        ]

        # 4. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        response_mask = [1] * len(output.token_ids)

        logger.warning("Begin to extract response")
        # 5. 取 response，替换前 prefix_len 个 token 为固定前缀
        ids = output.token_ids[: self.response_length]
        ids[: self.prefix_len] = prefix_token_ids[: len(ids)]
        logger.warning("get new response\n\n\n")

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=ids,
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output
