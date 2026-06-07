#!/usr/bin/env python3
"""Create a minimal synthetic GRPO dataset for verl training test.

Creates a parquet file with simple prompts for testing the PS-enabled
verl GRPO training pipeline. Includes reward_model with ground_truth
for math reward scoring.
"""
import pandas as pd
import os

output_dir = os.path.expanduser("~/rollout-prefix/data/synthetic_grpo")
os.makedirs(output_dir, exist_ok=True)

# Create simple prompts (math-style questions that work with any tokenizer)
# Also include ground_truth answers for reward computation
questions = [
    ("What is 2 + 3? Think step by step.", "5"),
    ("What is 7 * 8? Think step by step.", "56"),
    ("What is 15 - 9? Think step by step.", "6"),
    ("What is 100 / 5? Think step by step.", "20"),
    ("What is 3 + 12? Think step by step.", "15"),
    ("What is 4 * 6? Think step by step.", "24"),
    ("What is 20 - 7? Think step by step.", "13"),
    ("What is 50 / 10? Think step by step.", "5"),
    ("What is 9 + 8? Think step by step.", "17"),
    ("What is 5 * 3? Think step by step.", "15"),
    ("What is 17 - 4? Think step by step.", "13"),
    ("What is 144 / 12? Think step by step.", "12"),
    ("What is 6 + 7? Think step by step.", "13"),
    ("What is 8 * 9? Think step by step.", "72"),
    ("What is 30 - 11? Think step by step.", "19"),
    ("What is 200 / 4? Think step by step.", "50"),
]

prompts = [[{"role": "user", "content": q}] for q, _ in questions]
data_source = ["math"] * len(prompts)
reward_model = [{"ground_truth": gt} for _, gt in questions]

df = pd.DataFrame({
    "prompt": prompts,
    "data_source": data_source,
    "reward_model": reward_model,
})

train_path = os.path.join(output_dir, "train.parquet")
df.to_parquet(train_path)
print(f"Created {len(df)} samples at {train_path}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample reward_model: {df['reward_model'].iloc[0]}")