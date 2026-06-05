#!/usr/bin/env bash
# Setup Ray + verl + prefix-sharing environment
#
# This script starts a Ray cluster on the local machine and sets up the
# PYTHONPATH for verl and prefix-sharing. Run before using any Ray-based
# tests or verl training scripts.
#
# Usage:
#   source scripts/setup_ray_verl.sh
#
# After sourcing, these are available:
#   - Ray cluster running (ray status, ray dashboard at :8265)
#   - PYTHONPATH includes verl and prefix-sharing
#   - OMCONF_PATCHED=1 (omegaconf monkey-patch applied)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DEPS="$REPO_DIR/dependency"

# Activate conda if available
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate llm 2>/dev/null || true
fi

echo "=== Setting up Ray + verl + prefix-sharing ==="

# Stop any existing Ray instance
ray stop --force 2>/dev/null

# Start Ray head node with 8 GPUs
echo "Starting Ray cluster..."
ray start --head --port=6379 --num-cpus=64 --num-gpus=8 --disable-usage-stats 2>&1 | tail -3

# Verify Ray
python -c "import ray; ray.init(address='auto'); print(f'Ray OK: {ray.cluster_resources()}'); ray.shutdown()" 2>/dev/null && echo "Ray cluster: OK" || echo "Ray cluster: FAILED"

# Set PYTHONPATH for verl + prefix-sharing + megatron
export PYTHONPATH="$DEPS/verl_v070:$DEPS/megatron_v0150:$REPO_DIR/prefix-sharing:$PYTHONPATH"
echo "PYTHONPATH set (verl + megatron + prefix-sharing)"

# Verify imports
python -c "import verl; print(f'verl OK: {verl.__file__}')" 2>/dev/null || echo "verl: FAILED"
python -c "from prefix_sharing.core.model_spec import QWEN3_6_27B; print(f'PS OK: {QWEN3_6_27B.num_hidden_layers} layers')" 2>/dev/null || echo "PS: FAILED"

echo ""
echo "=== Environment ready ==="
echo "Ray dashboard: http://$(hostname -I | awk '{print $1}'):8265"
echo "To stop: ray stop"
echo ""
echo "Run tests:"
echo "  cd $REPO_DIR"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/gpu_e2e_verl_ps_ray.py"
