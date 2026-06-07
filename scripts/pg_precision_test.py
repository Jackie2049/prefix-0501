"""Quick precision validation test for PG block-causal mask."""
import os, gc, torch, torch.nn.functional as F, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from prefix_grouper import PrefixGrouper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

print("Loading model (SDPA)...")
model_path = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    attn_implementation="sdpa",
)
model.eval()

n = 4
prompt_length = 64
response_length = 64

# Create unpadded prompt
text = "What is 5+5? " + "Let me think step by step. First, I need to add these numbers. Addition is a fundamental arithmetic operation. I will compute the sum carefully. " * 3
ids = tokenizer.encode(text, add_special_tokens=True)
ids = ids[:prompt_length]
while len(ids) < prompt_length:
    ids.append(ids[-1])
prompts = torch.tensor([ids], dtype=torch.long, device=device)
prompt_mask = torch.ones_like(prompts)

# Generate n different responses
all_responses = []
for i in range(n):
    with torch.no_grad():
        gen = model.generate(
            prompts, attention_mask=prompt_mask,
            max_new_tokens=response_length, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    all_responses.append(gen[:, prompt_length:])
responses = torch.cat(all_responses, dim=0)  # (n, response_length)

# Normal mode: forward pass on each (prompt + response_i) independently
print("Computing normal logits...")
normal_logits_list = []
for i in range(n):
    full_seq = torch.cat([prompts[0:1], responses[i:i+1]], dim=1)
    full_mask = torch.ones_like(full_seq)
    with torch.no_grad():
        out = model(input_ids=full_seq, attention_mask=full_mask)
        normal_logits_list.append(out.logits.cpu())
normal_logits = torch.cat(normal_logits_list, dim=0)  # (n, seq_len, vocab)

# PG mode: grouped forward pass with block-causal mask
print("Computing PG logits...")
response_mask = torch.ones_like(responses)
for i in range(responses.size(0)):
    eos_pos = (responses[i] == tokenizer.eos_token_id).nonzero()
    if len(eos_pos) > 0:
        response_mask[i, eos_pos[0].item() + 1:] = 0

pg = PrefixGrouper.from_ungrouped_masks(
    prefix_mask=prompt_mask, suffix_mask=response_mask,
    group_sizes=[n], padding_mode="right", device=device,
)

concat_input_ids = pg.concat_input(prompts, prompt_mask, responses, response_mask)

# Build position_ids
group = pg.group_info[0]
prefix_len = group.prefix_len
pg_position_ids = torch.zeros(1, pg.padding_mask.size(1), dtype=torch.long, device=device)
pg_position_ids[0, :prefix_len] = torch.arange(prefix_len, device=device)
cur_pos = prefix_len
for s_len in group.suffix_lens:
    if s_len > 0:
        pg_position_ids[0, cur_pos:cur_pos + s_len] = torch.arange(prefix_len, prefix_len + s_len, device=device)
        cur_pos += s_len

# Build 4D block-causal mask (vectorized)
seq_len = pg.padding_mask.size(1)
causal = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0).bool()
causal = causal.unsqueeze(0).unsqueeze(0)

suffix_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)
cur_pos2 = prefix_len
for s_idx, s_len in enumerate(group.suffix_lens):
    if s_len > 0:
        suffix_id[0, cur_pos2:cur_pos2 + s_len] = s_idx + 1
        cur_pos2 += s_len

i_sid = suffix_id.unsqueeze(1).unsqueeze(2)
j_sid = suffix_id.unsqueeze(1).unsqueeze(3)
same_block = (i_sid == j_sid) | (j_sid == 0)
block_ok = same_block & causal

valid_q = pg.padding_mask.unsqueeze(1).unsqueeze(2).bool()
valid_kv = pg.padding_mask.unsqueeze(1).unsqueeze(3).bool()

# Prefix real-token mask
pos_valid = pg.padding_mask.bool()
pos_valid[0, :prefix_len] = prompt_mask[0].bool()
kv_valid = pos_valid.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
q_valid = pos_valid.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, seq_len)

mask_bool = block_ok & valid_q & valid_kv & kv_valid & q_valid

# Cast to bfloat16 (matching query dtype)
block_causal_mask = torch.where(
    mask_bool,
    torch.tensor(0.0, dtype=torch.bfloat16, device=device),
    torch.tensor(float('-inf'), dtype=torch.bfloat16, device=device)
)

print(f"  concat shape: {concat_input_ids.shape}, mask shape: {block_causal_mask.shape}")

with torch.no_grad():
    out = model(input_ids=concat_input_ids, attention_mask=block_causal_mask, position_ids=pg_position_ids)
    concat_logits = out.logits

split_result = pg.split_output(concat_logits)
prefix_logits, _, suffix_logits, _ = split_result

# Reconstruct per-sample logits
pg_logits_per_sample = []
for i in range(n):
    sample = torch.cat([prefix_logits[0], suffix_logits[i]], dim=0)
    pg_logits_per_sample.append(sample.cpu())
pg_logits = torch.stack(pg_logits_per_sample, dim=0)

# Compare response logits
resp_start = prompt_length - 1
normal_resp = normal_logits[:, resp_start:resp_start + response_length, :]
pg_resp = pg_logits[:, resp_start:resp_start + response_length, :]

min_len = min(normal_resp.size(1), pg_resp.size(1))
normal_resp = normal_resp[:, :min_len, :]
pg_resp = pg_resp[:, :min_len, :]

normal_lp = F.log_softmax(normal_resp.float(), dim=-1)
pg_lp = F.log_softmax(pg_resp.float(), dim=-1)

n_flat = normal_lp.reshape(-1)
p_flat = pg_lp.reshape(-1)

cos_sim = F.cosine_similarity(n_flat.unsqueeze(0), p_flat.unsqueeze(0)).item()
max_diff = (n_flat - p_flat).abs().max().item()
mean_diff = (n_flat - p_flat).abs().mean().item()

print(f"\nPRECISION RESULTS:")
print(f"  cos_sim = {cos_sim:.6f}")
print(f"  max_diff = {max_diff:.4f}")
print(f"  mean_diff = {mean_diff:.6f}")
status = "PASS" if cos_sim > 0.999 else "FAIL"
print(f"  Status: {status}")

# Also compare per-sample
for i in range(n):
    n_i = normal_lp[i].reshape(-1)
    p_i = pg_lp[i].reshape(-1)
    cs = F.cosine_similarity(n_i.unsqueeze(0), p_i.unsqueeze(0)).item()
    md = (n_i - p_i).abs().max().item()
    print(f"  Sample {i}: cos_sim={cs:.6f}, max_diff={md:.4f}")