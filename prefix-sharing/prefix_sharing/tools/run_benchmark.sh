#!/usr/bin/env bash
# ===========================================================================
# prefix-sharing 性能基准 runner
#
# 用同一份固定数据分别跑 PS-off 和 PS-on，采集 timing/memory CSV，
# 然后调用 analyze.py 产出对比 summary。
#
# 用法：
#   1. 填好下方 CONFIG 段的路径
#   2. bash prefix_sharing/tools/run_benchmark.sh
#
# 产出：
#   $RESULTS_DIR/ps_off/{timing,memory}_trace_rank*.csv
#   $RESULTS_DIR/ps_on/{timing,memory}_trace_rank*.csv
#   $RESULTS_DIR/ps_on/train.log            (含 [PS][audit] 日志)
#   $RESULTS_DIR/summary.txt                (analyze.py 输出)
# ===========================================================================
set -euo pipefail

# ── CONFIG（按你的环境修改）─────────────────────────────────────────────
# REPO_ROOT 留空 = 脚本从仓库根目录运行，所有路径相对解析
REPO_ROOT=""
PREFIX_SHARING_DIR="${REPO_ROOT}prefix-sharing"

# 模型 & verl 启动所需
MODEL_PATH="/path/to/Qwen2.5-0.5B"                      # 改成你的模型路径
TRAIN_PARQUET="/home/ma-user/work/tmp/data/512_gsm8k/train.parquet"
VAL_PARQUET="/home/ma-user/work/tmp/data/512_gsm8k/test.parquet"
CONFIG_NAME="ppo_megatron_trainer"

# 固定数据（仓库已提交，默认用 step-mode 合成数据）
FIXED_ROLLOUT="${PREFIX_SHARING_DIR}/prefix_sharing/tools/data/step_mode.json"
MANIFEST="${PREFIX_SHARING_DIR}/prefix_sharing/tools/data/manifest.json"

# 结果输出目录（REPO_ROOT 下）
RESULTS_DIR="${REPO_ROOT}benchmark_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 训练规模（小规模快速打通流程；上规模时调大）
TRAIN_BATCH_SIZE=96              # 与 step_mode.json 样本数对齐
MAX_PROMPT_LENGTH=320            # >= step_mode 最大序列长度
MAX_RESPONSE_LENGTH=32
PPO_MINI_BATCH_SIZE=96
PPO_MICRO_BATCH_SIZE_PER_GPU=96
TOTAL_TRAINING_STEPS=1
N_GPUS_PER_NODE=1
N_WORKERS=8                      # inject_fixed_rollout 的 num_workers，需整除样本数

# 并行策略（actor / ref 共用）
TP_SIZE=1
PP_SIZE=1
CP_SIZE=1
SEQUENCE_PARALLEL=False
# ────────────────────────────────────────────────────────────────────────

export PYTHONPATH="$PREFIX_SHARING_DIR:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0   # NPU

mkdir -p "$RESULTS_DIR/ps_off" "$RESULTS_DIR/ps_on"

# ── 单次训练运行 ────────────────────────────────────────────────────────
run_mode() {
    local enable_ps="$1"   # 0 or 1
    local profile_dir="$2"
    local log_path="$3"

    echo "============================================================"
    echo "  Running ENABLE_PREFIX_SHARING=$enable_ps"
    echo "  profile_dir = $profile_dir"
    echo "============================================================"

    rm -f "$profile_dir"/timing_trace_rank*.csv "$profile_dir"/memory_trace_rank*.csv

    ENABLE_PREFIX_SHARING="$enable_ps" \
    USE_FIXED_ROLLOUT="$FIXED_ROLLOUT" \
    PROFILE_OUTPUT_DIR="$profile_dir" \
    PYTHONUNBUFFERED=1 \
    python3 -m verl.trainer.main_ppo \
        --config-name="$CONFIG_NAME" \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_PARQUET" \
        data.val_files="$VAL_PARQUET" \
        data.train_batch_size="$TRAIN_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.megatron.use_remove_padding=True \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TP_SIZE" \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$PP_SIZE" \
        actor_rollout_ref.actor.megatron.context_parallel_size="$CP_SIZE" \
        actor_rollout_ref.actor.megatron.sequence_parallel="$SEQUENCE_PARALLEL" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n="$N_WORKERS" \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.ref.megatron.use_remove_padding=True \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$TP_SIZE" \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$PP_SIZE" \
        actor_rollout_ref.ref.megatron.context_parallel_size="$CP_SIZE" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=console \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes=1 \
        trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        2>&1 | tee "$log_path"
}

# ── 跑两次 ──────────────────────────────────────────────────────────────
run_mode 0 "$RESULTS_DIR/ps_off" "$RESULTS_DIR/ps_off/train.log"
run_mode 1 "$RESULTS_DIR/ps_on"  "$RESULTS_DIR/ps_on/train.log"

# ── 分析对比 ────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Analyzing results"
echo "============================================================"
python3 -m prefix_sharing.tools.analyze \
    --ps-off-dir "$RESULTS_DIR/ps_off" \
    --ps-on-dir  "$RESULTS_DIR/ps_on" \
    --manifest   "$MANIFEST" \
    --log-on     "$RESULTS_DIR/ps_on/train.log" \
    --json-out   "$RESULTS_DIR/summary_${TIMESTAMP}.json" \
    2>&1 | tee "$RESULTS_DIR/summary_${TIMESTAMP}.txt"

echo ""
echo "Done. Summary: $RESULTS_DIR/summary_${TIMESTAMP}.txt"
