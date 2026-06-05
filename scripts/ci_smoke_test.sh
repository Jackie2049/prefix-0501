#!/usr/bin/env bash
# CI Smoke Test: Run all prefix-sharing GPU tests and produce a pass/fail summary.
#
# Usage:
#   bash scripts/ci_smoke_test.sh            # all tests on GPU 0 (+ GPU 2 for TP=2)
#   CUDA_VISIBLE_DEVICES_TP=0,2 bash ...     # override TP device pair
#
# Requires: torchrun, pytest, flash-attn

set -o pipefail

# Activate conda environment if available
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate llm 2>/dev/null || true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

DEV="${CUDA_VISIBLE_DEVICES:-0}"
DEV_TP="${CUDA_VISIBLE_DEVICES_TP:-0,2}"

PASSED=0
FAILED=0
PASS_NAMES=()
FAIL_NAMES=()

_run() {
    local name="$1"; local nproc="$2"; local port="$3"; shift 3
    echo ""
    echo "=== $name ==="
    local out rc=0 f=0 c=0
    out=$(CUDA_VISIBLE_DEVICES="$DEV" torchrun --nproc_per_node="$nproc" --nnodes=1 --master_port="$port" "$@" 2>&1) || rc=$?
    c=$(echo "$out" | grep -c '\[PASS\]')
    f=$(echo "$out" | grep -c '\[FAIL\]')
    if [ "$rc" -eq 0 ] && [ "$f" -eq 0 ]; then
        PASSED=$((PASSED + 1)); PASS_NAMES+=("$name")
        echo "  PASS: $c checks" $([ "$rc" -ne 0 ] && echo "rc=$rc")
    else
        FAILED=$((FAILED + 1)); FAIL_NAMES+=("$name")
        echo "  FAIL: exit=$rc FAIL=$f PASS=$c"
        echo "$out" | grep -E 'FAIL|Error|Traceback' | tail -5
    fi
}

_run_tp() {
    local name="$1"; local port="$2"; shift 2
    echo ""
    echo "=== $name (TP=2) ==="
    local out rc=0 f=0 c=0
    out=$(CUDA_VISIBLE_DEVICES="$DEV_TP" torchrun --nproc_per_node=2 --nnodes=1 --master_port="$port" "$@" 2>&1) || rc=$?
    c=$(echo "$out" | grep -c '\[PASS\]')
    f=$(echo "$out" | grep -c '\[FAIL\]')
    if [ "$rc" -eq 0 ] && [ "$f" -eq 0 ]; then
        PASSED=$((PASSED + 1)); PASS_NAMES+=("$name")
        echo "  PASS: $c checks"
    else
        FAILED=$((FAILED + 1)); FAIL_NAMES+=("$name")
        echo "  FAIL: exit=$rc FAIL=$f PASS=$c"
        echo "$out" | grep -E 'FAIL|Error|Traceback' | tail -5
    fi
}

echo "============================================================"
echo "Prefix-Sharing CI Smoke Test"
echo "Repo: $REPO_DIR"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Single GPU: $DEV  |  TP=2 GPU: $DEV_TP"
echo "============================================================"

# pytest (direct python, no torchrun)
echo ""
echo "=== pytest ==="
pytest_out=$(CUDA_VISIBLE_DEVICES="$DEV" python -m pytest prefix-sharing/tests/ -x -q 2>&1)
pytest_rc=$?
pytest_p=$(echo "$pytest_out" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
pytest_f=$(echo "$pytest_out" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
if [ "$pytest_rc" -eq 0 ] && [ "$pytest_f" = "0" ]; then
    PASSED=$((PASSED + 1)); PASS_NAMES+=("pytest")
    echo "  PASS: $pytest_p passed"
else
    FAILED=$((FAILED + 1)); FAIL_NAMES+=("pytest")
    echo "  FAIL: $pytest_f failed"
fi

PORT=29520
_run   "GQA verify"              1 $((PORT++)) scripts/_gqa_verify.py
_run   "mcore verify"            1 $((PORT++)) scripts/gpu_mcore_verify.py
_run   "E2E mcore Qwen3.6"       1 $((PORT++)) scripts/e2e_mcore_qwen36.py
_run   "PS+mcore integration"    1 $((PORT++)) scripts/gpu_ps_mcore_integration.py
_run   "GDN PS E2E"              1 $((PORT++)) scripts/gpu_gdn_ps_e2e.py
_run   "verl GRPO sim"           1 $((PORT++)) scripts/gpu_verl_grpo_sim.py
_run   "numerical correctness"   1 $((PORT++)) scripts/gpu_e2e_numerical_correctness.py
_run   "perf benchmark"          1 $((PORT++)) scripts/gpu_ps_perf_benchmark.py
_run_tp "TP=2"                   $((PORT++)) scripts/gpu_tp_test.py

# Standalone python tests (no torchrun, need PYTHONPATH)
# Find python - prefer conda env python, fall back to system python
PY=$(which python 2>/dev/null || echo "$HOME/anaconda3/envs/llm/bin/python")

# Standalone run helper (no torchrun)
_run_standalone() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    local out rc=0
    out=$(CUDA_VISIBLE_DEVICES="$DEV" PYTHONPATH=prefix-sharing $PY "$@" 2>&1) || rc=$?
    local result_line
    result_line=$(echo "$out" | grep -E 'Results:')
    if [ "$rc" -eq 0 ] && echo "$result_line" | grep -q '\[PASS\]'; then
        PASSED=$((PASSED + 1)); PASS_NAMES+=("$name")
        local n
        n=$(echo "$result_line" | grep -oE '[0-9]+/' | tr -d '/')
        echo "  PASS: ${n:-?} checks"
    else
        FAILED=$((FAILED + 1)); FAIL_NAMES+=("$name")
        echo "  FAIL: exit=$rc  result=$result_line"
    fi
}

_run_standalone "hybrid attention" prefix-sharing/tests/run_hybrid_attention.py --device cuda
_run_standalone "deltanet state reuse" prefix-sharing/tests/run_deltanet.py --device cuda

# Summary
echo ""
echo "============================================================"
echo "CI Smoke Test Results"
echo "============================================================"
echo "Passed: $PASSED"
for n in "${PASS_NAMES[@]}"; do echo "  [PASS] $n"; done
echo "Failed: $FAILED"
for n in "${FAIL_NAMES[@]}"; do echo "  [FAIL] $n"; done
echo ""
TOTAL=$((PASSED + FAILED))
echo "Result: $PASSED/$TOTAL suites passed"
echo "============================================================"

[ "$FAILED" -gt 0 ] && exit 1 || exit 0
