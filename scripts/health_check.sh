#!/usr/bin/env bash
# Prefix-Sharing Health Check — quick validation of the entire environment.
#
# Usage: bash scripts/health_check.sh
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass=0; fail=0; warn=0

check() { local name="$1"; shift; if "$@" &>/dev/null; then echo -e "  ${GREEN}[PASS]${NC} $name"; pass=$((pass+1)); else echo -e "  ${RED}[FAIL]${NC} $name"; fail=$((fail+1)); fi; }
wcheck() { local name="$1"; shift; if "$@" &>/dev/null; then echo -e "  ${GREEN}[PASS]${NC} $name"; pass=$((pass+1)); else echo -e "  ${YELLOW}[WARN]${NC} $name (optional)"; warn=$((warn+1)); fi; }

echo "========================================"
echo "Prefix-Sharing Health Check"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Activate conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate llm 2>/dev/null || true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
DEPS="$REPO/dependency"
export PYTHONPATH="$DEPS/verl_v070:$DEPS/megatron_v0150:$REPO/prefix-sharing:$PYTHONPATH"

echo ""
echo "--- Python Environment ---"
check "python3" python3 --version
check "torch + CUDA" python3 -c "import torch; assert torch.cuda.is_available()"
check "flash-attn" python3 -c "import flash_attn" 2>/dev/null || \
  wcheck "flash-attn (optional)" python3 -c "import flash_attn"

echo ""
echo "--- Prefix-Sharing Core ---"
check "config" python3 -c "from prefix_sharing.core.config import PrefixSharingConfig"
check "planner" python3 -c "from prefix_sharing.core.planner import PrefixSharingPlanner"
check "detector" python3 -c "from prefix_sharing.core.prefix_detector import PrefixDetector"
check "model_spec" python3 -c "from prefix_sharing.core.model_spec import ModelSpec, QWEN3_6_27B"
check "batch_trim" python3 -c "from prefix_sharing.core.batch_trim import trim_inputs, trim_labels"
check "logprob" python3 -c "from prefix_sharing.core.logprob import restore_prefix_last_logprobs"

echo ""
echo "--- Backends ---"
check "torch_ref" python3 -c "from prefix_sharing.backends.torch_ref import TorchReferenceBackend"
check "packed_layout" python3 -c "from prefix_sharing.backends.packed_layout import PackedBatchLayout"
check "factory" python3 -c "from prefix_sharing.backends.factory import get_backend_instance"
wcheck "flash_atten_gpu" python3 -c "from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend"

echo ""
echo "--- Integrations ---"
check "context" python3 -c "from prefix_sharing.integrations.context import prefix_sharing_runtime_context"
check "verl_mcore" python3 -c "from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch, restore_suffix_first_log_probs_from_prefix"
check "megatron_runtime" python3 -c "from prefix_sharing.integrations.megatron_runtime import maybe_run_prefix_sharing_attention, maybe_run_prefix_sharing_deltanet"
check "patch_manager" python3 -c "from prefix_sharing.integrations.patch_manager import PatchManager"
check "megatron_attention" python3 -c "from prefix_sharing.integrations.megatron_attention import MegatronAttentionIntegration"

echo ""
echo "--- Megatron ---"
check "megatron.core" python3 -c "from megatron.core import parallel_state"
check "GatedDeltaNet" python3 -c "
import importlib.util,os
p=os.path.join('$DEPS','verl_v070','verl','models','mcore','gated_delta_net.py')
s=importlib.util.spec_from_file_location('x',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print('OK')"

echo ""
echo "--- verl ---"
wcheck "verl package" python3 -c "import verl" 2>/dev/null || \
  python3 -c "__import__('sys').path.insert(0,'$DEPS/verl_v070');import verl"

echo ""
echo "--- Ray (optional) ---"
wcheck "ray cluster" python3 -c "import ray; ray.init(address='auto',ignore_reinit_error=True); print(ray.cluster_resources()); ray.shutdown()"

echo ""
echo "--- Qwen3.6 Model ---"
check "QWEN3_6_27B layers" python3 -c "
from prefix_sharing.core.model_spec import QWEN3_6_27B
assert QWEN3_6_27B.num_hidden_layers == 64
assert QWEN3_6_27B.num_full_attention_layers == 16
assert QWEN3_6_27B.num_linear_attention_layers == 48
assert QWEN3_6_27B.num_attention_heads == 24
assert QWEN3_6_27B.num_key_value_heads == 4"

echo ""
echo "--- GPU Memory ---"
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    m=torch.cuda.mem_get_info(i)
    print(f'  GPU {i}: {p.name}, {p.total_memory//1024**3}GB, free={m[0]//1024**3}GB')" 2>/dev/null

echo ""
echo "========================================"
total=$((pass + fail + warn))
echo "Results: $pass passed, $fail failed, $warn warnings ($total total)"
[ $fail -eq 0 ] && echo -e "${GREEN}HEALTHY${NC}" || echo -e "${RED}ISSUES FOUND${NC}"
echo "========================================"
exit $fail
