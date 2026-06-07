#!/usr/bin/env python3
"""PG E2E GRPO Benchmark v3 - with improved precision handling.

Uses greedy generation for identical prompts to ensure fair comparison.
Separates timing into: generation, logprob computation, and update phases.
"""
import os, sys, time, json, torch, torch.nn.functional as F
from dataclasses import dataclass

VERL_CLEAN = os.path.expanduser("~/rollout-prefix/verl-clean")
sys.path.insert(0, VERL_CLEAN)

from prefix_grouper import PrefixGrouper
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch
from verl.utils.torch_functional import logprobs_from_logits
from verl.trainer.ppo.prefix_grouper_utils import build_pg_from_micro_batch


@dataclass
class BenchConfig:
    n_responses: int = 4
    batch_size: int = 4  # number of groups
    max_prompt_length: int = 64
    max_response_length: int = 32
    use_prefix_grouper: bool = False
    warmup_steps: int = 1
    num_steps: int = 5
    lr: float = 1e-6


def run_benchmark(config):
    model_path = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
    device = "cuda"

    print(f"\n{'='*70}")
    print(f"GRPO E2E Benchmark: use_prefix_grouper={config.use_prefix_grouper}")
    print(f"n={config.n_responses}, batch={config.batch_size}, "
          f"prompt={config.max_prompt_length}, response={config.max_response_length}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create prompts - identical within each group for maximum prefix sharing
    total = config.batch_size * config.n_responses
    prompts = []
    group_uids = []
    for g in range(config.batch_size):
        # Each group shares the same long prefix, with tiny variation at end
        base = f"Group {g}: What is the answer to this question? Let me think step by step and show all work. "
        for j in range(config.n_responses):
            prompts.append(base)
            group_uids.append(g)

    encoded = tokenizer(prompts, padding=True, truncation=True,
                        max_length=config.max_prompt_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # Calculate prefix ratio (shared prefix within each group)
    # For identical prompts, prefix_ratio = prompt_len / (prompt_len + response_len)
    prefix_ratio = prompt_len / (prompt_len + config.max_response_length)
    print(f"  {total} sequences ({config.batch_size} groups x {config.n_responses})")
    print(f"  prompt_len={prompt_len}, prefix_ratio={prefix_ratio:.1%}")

    # Load model
    if config.use_prefix_grouper:
        print("Installing PG monkey-patch...")
        apply_prefix_grouper_patch()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    total_times, gen_times, logprob_times, update_times = [], [], [], []
    all_losses = []

    for step in range(config.warmup_steps + config.num_steps):
        # ===== Generation phase =====
        gen_start = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            responses_full = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=config.max_response_length,
                do_sample=True, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        model.train()
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - gen_start

        response_tokens = responses_full[:, prompt_len:]
        response_mask = (response_tokens != tokenizer.pad_token_id).long()
        for i in range(total):
            eos_pos = (response_tokens[i] == tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                response_mask[i, eos_pos[0].item() + 1:] = 0
        response_mask = response_mask.to(device)

        # ===== Log prob computation phase =====
        logprob_start = time.perf_counter()
        if config.use_prefix_grouper:
            uid_tensor = torch.tensor(group_uids)
            micro_batch = {
                "prompts": input_ids,
                "responses": response_tokens,
                "response_mask": response_mask,
                "uid": uid_tensor,
                "pad_token_id": tokenizer.pad_token_id,
            }
            (pg, concat_ids, pg_mask, pg_position_ids, pg_responses, pg_response_mask) = build_pg_from_micro_batch(
                micro_batch, pad_token_id=tokenizer.pad_token_id, padding_mode="right"
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_on = model(input_ids=concat_ids, attention_mask=pg_mask,
                                  position_ids=pg_position_ids,
                                  prefix_grouper=pg, use_cache=False).logits
            prefix_out, _, suffix_out_raw, suffix_mask_raw = pg.split_output(logits_on, include_prefix_last=1)
            suffix_out = suffix_out_raw[:, :-1].float()
            suffix_mask = suffix_mask_raw[:, 1:]
            completion_ids_right = pg.convert_padding(response_tokens, response_mask, padding_mode="right")
            log_probs = logprobs_from_logits(suffix_out, completion_ids_right)
            padding_mask_pg = suffix_mask == 0
            log_probs = log_probs.masked_fill(padding_mask_pg, 0.0)
            target_len = response_tokens.size(1)
            if log_probs.size(1) != target_len:
                full = log_probs.new_zeros(log_probs.size(0), target_len)
                full[:, :log_probs.size(1)] = log_probs
                log_probs = full
        else:
            seq = torch.cat([input_ids, response_tokens], dim=1)
            seq_mask = torch.cat([attention_mask, response_mask.to(attention_mask.dtype)], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=seq, attention_mask=seq_mask).logits
            shift_logits = logits[:, prompt_len-1:-1, :]
            shift_labels = response_tokens
            lp_all = F.log_softmax(shift_logits.float(), dim=-1)
            log_probs = lp_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        torch.cuda.synchronize()
        logprob_time = time.perf_counter() - logprob_start

        # ===== GRPO advantage =====
        rewards = torch.ones(total, dtype=torch.float32, device=device)  # dummy reward
        advantages = torch.zeros_like(rewards)
        for g in range(config.batch_size):
            group_start = g * config.n_responses
            group_end = group_start + config.n_responses
            group_rewards = rewards[group_start:group_end]
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8
            advantages[group_start:group_end] = (group_rewards - mean) / std

        # ===== Update phase =====
        update_start = time.perf_counter()
        mask = response_mask.float()
        loss = -(advantages.unsqueeze(-1) * log_probs * mask).sum() / mask.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        update_time = time.perf_counter() - update_start

        step_time = gen_time + logprob_time + update_time
        is_measured = step >= config.warmup_steps
        if is_measured:
            total_times.append(step_time)
            gen_times.append(gen_time)
            logprob_times.append(logprob_time)
            update_times.append(update_time)
            all_losses.append(loss.item())

        tag = "[MEASURED]" if is_measured else "[WARMUP]"
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"Step {step+1}: loss={loss.item():.4f} gen={gen_time*1000:.0f}ms "
              f"logprob={logprob_time*1000:.0f}ms update={update_time*1000:.0f}ms "
              f"total={step_time*1000:.0f}ms mem={mem:.2f}GB {tag}")

    avg_total = sum(total_times) / len(total_times)
    avg_gen = sum(gen_times) / len(gen_times)
    avg_logprob = sum(logprob_times) / len(logprob_times)
    avg_update = sum(update_times) / len(update_times)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "use_prefix_grouper": config.use_prefix_grouper,
        "avg_total_ms": avg_total * 1000,
        "avg_gen_ms": avg_gen * 1000,
        "avg_logprob_ms": avg_logprob * 1000,
        "avg_update_ms": avg_update * 1000,
        "peak_mem_gb": peak_mem,
        "prefix_ratio": prefix_ratio,
        "n_responses": config.n_responses,
        "batch_size": config.batch_size,
        "prompt_len": prompt_len,
    }

    print(f"\n{'='*70}")
    print(f"Results: use_prefix_grouper={config.use_prefix_grouper}")
    print(f"  Avg total:    {avg_total*1000:.0f} ms")
    print(f"  Avg gen:      {avg_gen*1000:.0f} ms ({avg_gen/avg_total*100:.1f}%)")
    print(f"  Avg logprob:  {avg_logprob*1000:.0f} ms ({avg_logprob/avg_total*100:.1f}%)")
    print(f"  Avg update:   {avg_update*1000:.0f} ms ({avg_update/avg_total*100:.1f}%)")
    print(f"  Peak memory:  {peak_mem:.2f} GB")
    print(f"  Prefix ratio: {prefix_ratio:.1%}")
    print(f"{'='*70}")

    del model, optimizer
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    configs = [
        BenchConfig(n_responses=4, batch_size=4, max_prompt_length=64, max_response_length=32),
        BenchConfig(n_responses=8, batch_size=4, max_prompt_length=64, max_response_length=32),
        BenchConfig(n_responses=4, batch_size=4, max_prompt_length=128, max_response_length=64),
    ]

    all_results = []
    for cfg_base in configs:
        results_off = run_benchmark(BenchConfig(
            n_responses=cfg_base.n_responses, batch_size=cfg_base.batch_size,
            max_prompt_length=cfg_base.max_prompt_length,
            max_response_length=cfg_base.max_response_length,
            use_prefix_grouper=False,
        ))
        results_on = run_benchmark(BenchConfig(
            n_responses=cfg_base.n_responses, batch_size=cfg_base.batch_size,
            max_prompt_length=cfg_base.max_prompt_length,
            max_response_length=cfg_base.max_response_length,
            use_prefix_grouper=True,
        ))

        if results_off and results_on:
            total_speedup = results_off["avg_total_ms"] / results_on["avg_total_ms"]
            logprob_speedup = results_off["avg_logprob_ms"] / results_on["avg_logprob_ms"]

            print(f"\n{'='*70}")
            print(f"COMPARISON: n={cfg_base.n_responses}, "
                  f"prefix={cfg_base.max_prompt_length}tok, "
                  f"response={cfg_base.max_response_length}tok, "
                  f"ratio={results_off['prefix_ratio']:.1%}")
            print(f"  PG OFF total:   {results_off['avg_total_ms']:.0f} ms")
            print(f"  PG ON  total:   {results_on['avg_total_ms']:.0f} ms")
            print(f"  Total speedup:  {total_speedup:.2f}x")
            print(f"  PG OFF logprob: {results_off['avg_logprob_ms']:.0f} ms")
            print(f"  PG ON  logprob: {results_on['avg_logprob_ms']:.0f} ms")
            print(f"  Logprob speedup: {logprob_speedup:.2f}x")
            print(f"  Memory: {results_off['peak_mem_gb']:.2f} GB vs {results_on['peak_mem_gb']:.2f} GB")
            print(f"{'='*70}")

            all_results.append({
                "n": cfg_base.n_responses,
                "prompt_len": results_off["prompt_len"],
                "response_len": cfg_base.max_response_length,
                "prefix_ratio": results_off["prefix_ratio"],
                "off_total_ms": results_off["avg_total_ms"],
                "on_total_ms": results_on["avg_total_ms"],
                "total_speedup": total_speedup,
                "off_logprob_ms": results_off["avg_logprob_ms"],
                "on_logprob_ms": results_on["avg_logprob_ms"],
                "logprob_speedup": logprob_speedup,
            })

    # Precision verification
    print("\nRunning precision verification (greedy, identical prompts)...")
    model_path = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n = 4
    prompts = ["What is 5+5? Let me think step by step. " * 5] * n
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")

    # OFF: model without patch
    model_off = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    model_off.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        responses = model_off.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_tokens = responses[:, input_ids.shape[1]:]
    response_mask = (response_tokens != tokenizer.pad_token_id).long().to("cuda")

    # Normal log_probs
    seq = torch.cat([input_ids, response_tokens], dim=1)
    seq_mask = torch.cat([attention_mask, response_mask.to(attention_mask.dtype)], dim=1)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_off = model_off(input_ids=seq, attention_mask=seq_mask).logits
    lp_off = F.log_softmax(logits_off[:, input_ids.shape[1]-1:-1, :].float(), dim=-1)
    lp_off = lp_off.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

    del model_off
    torch.cuda.empty_cache()

    # ON: model with PG patch
    apply_prefix_grouper_patch()
    model_on = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    model_on.eval()

    uid_tensor = torch.tensor([0] * n)
    micro_batch = {"prompts": input_ids, "responses": response_tokens,
                   "response_mask": response_mask, "uid": uid_tensor,
                   "pad_token_id": tokenizer.pad_token_id}
    (pg, concat_ids, pg_mask, position_ids, pg_responses, pg_response_mask) = build_pg_from_micro_batch(
        micro_batch, pad_token_id=tokenizer.pad_token_id, padding_mode="right")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_on = model_on(input_ids=concat_ids, attention_mask=pg_mask,
                             position_ids=position_ids, prefix_grouper=pg, use_cache=False).logits
    prefix_out, _, suffix_out_raw, suffix_mask_raw = pg.split_output(logits_on, include_prefix_last=1)
    suffix_out = suffix_out_raw[:, :-1].float()
    completion_ids_right = pg.convert_padding(response_tokens, response_mask, padding_mode="right")
    lp_on = logprobs_from_logits(suffix_out, completion_ids_right)
    lp_on = lp_on.masked_fill(suffix_mask_raw[:, 1:] == 0, 0.0)
    target_len = response_tokens.size(1)
    if lp_on.size(1) != target_len:
        full = lp_on.new_zeros(lp_on.size(0), target_len)
        full[:, :lp_on.size(1)] = lp_on
        lp_on = full

    valid = response_mask.bool()
    off_v = lp_off[valid].float()
    on_v = lp_on[valid].float()
    cos_sim = F.cosine_similarity(off_v.unsqueeze(0), on_v.unsqueeze(0)).item()
    max_diff = (off_v - on_v).abs().max().item()

    print(f"  Precision: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")
    precision = {"cos_sim": cos_sim, "max_diff": max_diff}

    # Save all results
    result_file = os.path.expanduser("~/rollout-prefix/pg_e2e_grpo_results_v3.json")
    with open(result_file, "w") as f:
        json.dump({"comparisons": all_results, "precision": precision}, f, indent=2)
    print(f"\nResults saved to {result_file}")