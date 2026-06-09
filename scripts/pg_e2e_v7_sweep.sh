#!/bin/bash
# PG E2E GRPO Competitive Analysis — v7 Multi-Config Sweep
# Runs PG OFF and PG ON with N=2,4,8 and P_LEN=64,128
# Single GPU (GPU 0), Qwen2.5-0.5B-Instruct, colocate mode

set -e

source ~/anaconda3/bin/activate llm
export PYTHONPATH=/home/zxw/rollout-prefix/verl-pg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

SCRIPT=/home/zxw/rollout-prefix/pg_e2e_grpo_v7_sweep.py
MODEL=/home/zxw/rollout-prefix/models/Qwen2.5-0.5B-Instruct
DATA=/home/zxw/rollout-prefix/data/grpo_math/train.parquet

echo "========================================"
echo "PG E2E GRPO v7 Multi-Config Sweep"
echo "========================================"

# ── PG OFF: N=2, P=64 ──
echo ">>> PG OFF, N=2, P_LEN=64"
python3 $SCRIPT USE_PG=0 N=2 P_LEN=64 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_off_N2_P64.txt

# ── PG ON: N=2, P=64 ──
echo ">>> PG ON, N=2, P_LEN=64"
python3 $SCRIPT USE_PG=1 N=2 P_LEN=64 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_on_N2_P64.txt

# ── PG OFF: N=4, P=64 ──
echo ">>> PG OFF, N=4, P_LEN=64"
python3 $SCRIPT USE_PG=0 N=4 P_LEN=64 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_off_N4_P64.txt

# ── PG ON: N=4, P=64 ──
echo ">>> PG ON, N=4, P_LEN=64"
python3 $SCRIPT USE_PG=1 N=4 P_LEN=64 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_on_N4_P64.txt

# ── PG OFF: N=8, P=64 ──
echo ">>> PG OFF, N=8, P_LEN=64"
python3 $SCRIPT USE_PG=0 N=8 P_LEN=64 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_off_N8_P64.txt

# ── PG ON: N=8, P=64 ── (may OOM)
echo ">>> PG ON, N=8, P_LEN=64"
python3 $SCRIPT USE_PG=1 N=8 P_LEN=64 NUM_STEPS=3 BATCH_SIZE=2 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_on_N8_P64.txt || echo "PG ON N=8 OOM — skipping"

# ── PG OFF: N=4, P=128 ──
echo ">>> PG OFF, N=4, P_LEN=128"
python3 $SCRIPT USE_PG=0 N=4 P_LEN=128 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_off_N4_P128.txt

# ── PG ON: N=4, P=128 ──
echo ">>> PG ON, N=4, P_LEN=128"
python3 $SCRIPT USE_PG=1 N=4 P_LEN=128 NUM_STEPS=5 BATCH_SIZE=4 MODEL_PATH=$MODEL DATA_PATH=$DATA GPU_ID=0 2>&1 | tee /home/zxw/rollout-prefix/pg_v7_log_on_N4_P128.txt

echo "========================================"
echo "All sweeps complete!"
echo "========================================"

# ── Collect results ──
echo ""
echo "=== RESULTS SUMMARY ==="
for f in /home/zxw/rollout-prefix/pg_v7_results_*.json; do
    echo ""
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
    pg = 'ON' if d['use_pg'] else 'OFF'
    print(f\"PG={pg} N={d['n']} P={d['p_len']}: step={d['avg_step_time']:.3f}s rollout={d['avg_rollout_time']:.3f}s({d['rollout_pct']:.1f}%) logprob={d['avg_logprob_time']:.3f}s({d['logprob_pct']:.1f}%) train={d['avg_train_time']:.3f}s({d['train_pct']:.1f}%) mem={d['avg_peak_mem']:.2f}GB\")
" 2>/dev/null || echo "Failed to parse $f"
done