#!/usr/bin/env python3
"""
PG E2E GRPO Competitive Analysis — v7 Multi-Config Sweep
Sweeps N (2,4,8) and P_LEN (64,128) to compare PG OFF vs PG ON.
Uses verl's PrefixGrouper monkey-patch for PG ON.
Single GPU, HF rollout (colocate mode).
"""

import os, sys, time, json, uuid, copy
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/zxw/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
DATA_PATH  = os.environ.get("DATA_PATH", "/home/zxw/rollout-prefix/data/grpo_math/train.parquet")
USE_PG     = os.environ.get("USE_PG", "0") == "1"
N          = int(os.environ.get("N", "4"))
P_LEN      = int(os.environ.get("P_LEN", "64"))
R_LEN      = int(os.environ.get("R_LEN", "128"))
NUM_STEPS  = int(os.environ.get("NUM_STEPS", "5"))
LR         = float(os.environ.get("LR", "1e-6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
SEED       = 42
GPU_ID     = os.environ.get("GPU_ID", "0")
DEVICE     = f"cuda:{GPU_ID}"
MAX_LEN    = P_LEN + R_LEN

# ── Apply PG monkey-patch if USE_PG ──
if USE_PG:
    sys.path.insert(0, "/home/zxw/rollout-prefix/verl-pg")
    from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch
    apply_prefix_grouper_patch()
    from prefix_grouper import PrefixGrouper
    from verl.trainer.ppo.prefix_grouper_utils import (
        build_pg_from_micro_batch, pg_forward, build_position_ids_for_prefix_grouper
    )
    print("[PG ON] PrefixGrouper monkey-patch applied", flush=True)


def load_data():
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    prompts = []
    for i in range(min(BATCH_SIZE, len(df))):
        p = df.iloc[i]["prompt"]
        if isinstance(p, np.ndarray):
            p = list(p)
        if isinstance(p, list):
            prompts.append(p)
        elif isinstance(p, str):
            try:
                prompts.append(eval(p))
            except:
                prompts.append([{"role": "user", "content": p}])
    return prompts


def compute_log_probs_normal(model, input_ids, attention_mask, prompt_lens):
    """Standard forward — no prefix sharing."""
    with torch.no_grad() if not model.training else nullcontext():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs_list = []
    for i in range(input_ids.shape[0]):
        p_len = prompt_lens[i]
        resp_logits = logits[i, p_len-1:-1, :].float()
        resp_tokens = input_ids[i, p_len:]
        max_resp = min(resp_logits.size(0), R_LEN)
        resp_logits_sliced = resp_logits[:max_resp]
        resp_tokens_sliced = resp_tokens[:max_resp]
        log_p = F.log_softmax(resp_logits_sliced, dim=-1)
        tok_lp = log_p.gather(dim=-1, index=resp_tokens_sliced.unsqueeze(-1)).squeeze(-1)
        padded = torch.zeros(R_LEN, device=DEVICE, dtype=tok_lp.dtype)
        padded[:tok_lp.size(0)] = tok_lp
        log_probs_list.append(padded)
    return torch.stack(log_probs_list)


def compute_log_probs_pg(model, input_ids, attention_mask, prompt_lens, uids, tokenizer):
    """PrefixGrouper forward — shared-prefix attention decomposition."""
    batch_size = input_ids.shape[0]
    pad_token_id = tokenizer.pad_token_id or 0

    response_mask = torch.zeros_like(attention_mask)
    for i in range(batch_size):
        response_mask[i, prompt_lens[i]:] = attention_mask[i, prompt_lens[i]:]

    micro_batch = {
        "prompts": input_ids,
        "responses": input_ids,
        "response_mask": response_mask,
        "uid": uids,
        "pad_token_id": pad_token_id,
    }

    pg, concat_input_ids, pg_attn_mask, position_ids, responses_pg, response_mask_pg = build_pg_from_micro_batch(
        micro_batch, pad_token_id=pad_token_id, padding_mode="right"
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        log_probs_pg, _, suffix_mask = pg_forward(
            model=model,
            prefix_grouper=pg,
            concat_input_ids=concat_input_ids,
            attention_mask=pg_attn_mask,
            position_ids=position_ids,
            completion_ids=input_ids,
            completion_mask=response_mask,
            temperature=1.0,
            padding_mode="right",
            include_prefix_last=1,
            calculate_entropy=False,
        )

    padding_mask = suffix_mask == 0
    log_probs_pg = log_probs_pg.masked_fill(padding_mask, 0.0)

    if log_probs_pg.size(1) < R_LEN:
        full_lp = log_probs_pg.new_zeros(batch_size, R_LEN)
        full_lp[:, :log_probs_pg.size(1)] = log_probs_pg
        log_probs_pg = full_lp

    return log_probs_pg


def main():
    torch.manual_seed(SEED)
    print(flush=True)
    print("=== PG E2E GRPO Competitive Analysis v7 (Multi-Config Sweep) ===", flush=True)
    print(f"USE_PG={USE_PG} (0=OFF, 1=ON)", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"N={N}, P_LEN={P_LEN}, R_LEN={R_LEN}, BATCH={BATCH_SIZE}, STEPS={NUM_STEPS}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn_impl = "flash_attention_2" if USE_PG else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation=attn_impl).to(DEVICE)
    model.train()

    # Reference model
    print("Loading ref model...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Load data
    print("Loading data...", flush=True)
    prompts = load_data()
    print(f"Loaded {len(prompts)} prompts", flush=True)

    prompt_texts = []
    prompt_lens_list = []
    for p in prompts:
        chat = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(chat)
        ids = tokenizer.encode(chat)
        prompt_lens_list.append(min(len(ids), P_LEN))

    results = []
    for step in range(NUM_STEPS):
        print(f"\n--- Step {step+1}/{NUM_STEPS} ---", flush=True)
        step_t0 = time.time()

        # Phase 1: Rollout
        rollout_t0 = time.time()
        all_input_ids = []
        all_attn_mask = []
        all_prompt_lens = []
        all_uids = []

        for pi, (chat, p_len) in enumerate(zip(prompt_texts, prompt_lens_list)):
            uid = str(uuid.uuid4())
            for j in range(N):
                inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=P_LEN)
                inp_ids = inp["input_ids"].to(DEVICE)
                inp_mask = inp["attention_mask"].to(DEVICE)
                actual_p_len = inp_ids.size(1)

                with torch.no_grad():
                    gen_out = model.generate(
                        inp_ids,
                        attention_mask=inp_mask,
                        max_new_tokens=R_LEN,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                full_ids = gen_out[0]
                full_len = full_ids.size(0)

                padded = torch.full((MAX_LEN,), tokenizer.pad_token_id, dtype=torch.long)
                padded[:full_len] = full_ids[:MAX_LEN]
                mask = torch.zeros(MAX_LEN, dtype=torch.long)
                mask[:min(full_len, MAX_LEN)] = 1

                all_input_ids.append(padded)
                all_attn_mask.append(mask)
                all_prompt_lens.append(actual_p_len)
                all_uids.append(uid)

        input_ids = torch.stack(all_input_ids).to(DEVICE)
        attn_mask = torch.stack(all_attn_mask).to(DEVICE)
        rollout_time = time.time() - rollout_t0
        print(f"  Rollout: {rollout_time:.3f}s ({len(all_input_ids)} seqs, N={N})", flush=True)

        # Phase 2: Log probs
        logprob_t0 = time.time()
        if USE_PG:
            log_probs = compute_log_probs_pg(model, input_ids, attn_mask, all_prompt_lens, all_uids, tokenizer)
        else:
            log_probs = compute_log_probs_normal(model, input_ids, attn_mask, all_prompt_lens)

        # Ref log probs
        with torch.no_grad():
            ref_out = ref_model(input_ids=input_ids, attention_mask=attn_mask)
            ref_logits = ref_out.logits
            ref_log_probs = []
            for i in range(input_ids.shape[0]):
                p_len = all_prompt_lens[i]
                resp_logits = ref_logits[i, p_len-1:-1, :].float()
                resp_tokens = input_ids[i, p_len:]
                max_resp = min(resp_logits.size(0), R_LEN)
                resp_logits_sliced = resp_logits[:max_resp]
                resp_tokens_sliced = resp_tokens[:max_resp]
                log_p = F.log_softmax(resp_logits_sliced, dim=-1)
                tok_lp = log_p.gather(dim=-1, index=resp_tokens_sliced.unsqueeze(-1)).squeeze(-1)
                padded = torch.zeros(R_LEN, device=DEVICE, dtype=tok_lp.dtype)
                padded[:tok_lp.size(0)] = tok_lp
                ref_log_probs.append(padded)
            ref_log_probs = torch.stack(ref_log_probs)

        logprob_time = time.time() - logprob_t0
        print(f"  LogProb: {logprob_time:.3f}s", flush=True)

        # Phase 3: Advantages
        adv_t0 = time.time()
        advantages = torch.zeros_like(log_probs)
        for uid in set(all_uids):
            indices = [i for i, u in enumerate(all_uids) if u == uid]
            group_lp = log_probs[indices]
            mean_lp = group_lp.mean()
            advantages[indices] = group_lp - mean_lp

        resp_mask = torch.zeros_like(log_probs)
        for i in range(input_ids.shape[0]):
            p_len = all_prompt_lens[i]
            actual_len = int(attn_mask[i].sum().item())
            resp_len = min(actual_len - p_len, R_LEN)
            if resp_len > 0:
                resp_mask[i, :resp_len] = 1.0

        adv_time = time.time() - adv_t0

        # Phase 4: Loss + backward
        train_t0 = time.time()
        kl = log_probs - ref_log_probs
        loss = -(advantages * log_probs) + 0.001 * kl
        loss = (loss * resp_mask).sum() / resp_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_time = time.time() - train_t0
        step_time = time.time() - step_t0

        # Compute cos_sim between PG/normal log_probs for precision check
        # (only meaningful when both are computed)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result = {
            "step": step + 1,
            "step_time": step_time,
            "rollout_time": rollout_time,
            "logprob_time": logprob_time,
            "adv_time": adv_time,
            "train_time": train_time,
            "loss": loss.item(),
            "peak_mem_gb": peak_mem,
            "n_seqs": len(all_input_ids),
            "n_per_prompt": N,
            "p_len": P_LEN,
        }
        results.append(result)
        print(f"  Advantages: {adv_time:.3f}s", flush=True)
        print(f"  Train: {train_time:.3f}s", flush=True)
        print(f"  Step total: {step_time:.3f}s  (rollout {rollout_time/step_time*100:.0f}% | logprob {logprob_time/step_time*100:.0f}% | train {train_time/step_time*100:.0f}%)", flush=True)
        print(f"  Loss: {loss.item():.6f}", flush=True)
        print(f"  Peak GPU mem: {peak_mem:.2f} GB", flush=True)

    # Summary
    if len(results) > 1:
        avg_step = sum(r['step_time'] for r in results[1:]) / (len(results)-1)
        avg_rollout = sum(r['rollout_time'] for r in results[1:]) / (len(results)-1)
        avg_logprob = sum(r['logprob_time'] for r in results[1:]) / (len(results)-1)
        avg_train = sum(r['train_time'] for r in results[1:]) / (len(results)-1)
        avg_peak_mem = sum(r['peak_mem_gb'] for r in results[1:]) / (len(results)-1)
    else:
        avg_step = results[0]['step_time']
        avg_rollout = results[0]['rollout_time']
        avg_logprob = results[0]['logprob_time']
        avg_train = results[0]['train_time']
        avg_peak_mem = results[0]['peak_mem_gb']

    print("\n=== SUMMARY ===", flush=True)
    print(f"USE_PG={USE_PG} ({'ON' if USE_PG else 'OFF'})", flush=True)
    print(f"N={N}, P_LEN={P_LEN}, BATCH={BATCH_SIZE}", flush=True)
    print(f"Avg step:     {avg_step:.3f}s", flush=True)
    print(f"Avg rollout:  {avg_rollout:.3f}s ({avg_rollout/avg_step*100:.1f}%)", flush=True)
    print(f"Avg logprob:  {avg_logprob:.3f}s ({avg_logprob/avg_step*100:.1f}%)", flush=True)
    print(f"Avg train:    {avg_train:.3f}s ({avg_train/avg_step*100:.1f}%)", flush=True)
    print(f"Avg peak_mem: {avg_peak_mem:.2f} GB", flush=True)

    output_file = f"/home/zxw/rollout-prefix/pg_v7_results_{'pg_on' if USE_PG else 'pg_off'}_N{N}_P{P_LEN}.json"
    with open(output_file, "w") as f:
        json.dump({
            "use_pg": USE_PG,
            "n": N, "p_len": P_LEN, "r_len": R_LEN,
            "batch_size": BATCH_SIZE, "num_steps": NUM_STEPS,
            "results": results,
            "avg_step_time": avg_step,
            "avg_rollout_time": avg_rollout,
            "avg_logprob_time": avg_logprob,
            "avg_train_time": avg_train,
            "avg_peak_mem": avg_peak_mem,
            "rollout_pct": avg_rollout / avg_step * 100,
            "logprob_pct": avg_logprob / avg_step * 100,
            "train_pct": avg_train / avg_step * 100,
        }, f, indent=2)
    print(f"Results saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()