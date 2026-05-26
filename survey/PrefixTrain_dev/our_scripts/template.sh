#!/usr/bin/env bash
set -e
ROOT_PATH=$(pwd)
cd $ROOT_PATH/runtime
LOG_PATH=${ROOT_PATH}/logs/
LOG_PATH_=$LOG_PATH/
LOG_PREFIX="run"
########## Default values ##########
DP_SIMI=""
RECORD_PATH=""
RECORD_PREFIX=""
INPUT_TOKEN_PATH=""
EXTRA_ARGS=""
export ENABLE_WEIGHT_SHARE=0

########## Parse args ##########
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --dp-simi)
      DP_SIMI="$2"
      shift 2
      ;;
    --record-path)
      RECORD_PATH="$2"
      LOG_PATH_=$2  
      LOG_PATH=$2
      shift 2
      ;;
    --record-prefix)  
      RECORD_PREFIX="$2"
      LOG_PREFIX=$2
      shift 2
      ;;
    --input-token-path)
      INPUT_TOKEN_PATH="$2"
      shift 2
      ;;
    --share-activation|--enable-token-level-balance | --dp-workload-balance | --save-all-activation)
      EXTRA_ARGS+=" $1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

########## Required args ##########
if [[ -z "$CONFIG_NAME" ]]; then
  echo "Error: --config is required"
  exit 1
fi

########## Build optional args ##########
OPTIONAL_ARGS=""

[[ -n "$DP_SIMI" ]] && OPTIONAL_ARGS+=" --dp-simi $DP_SIMI"
[[ -n "$INPUT_TOKEN_PATH" ]] && OPTIONAL_ARGS+=" --input-token-path $INPUT_TOKEN_PATH"
[[ -n "$RECORD_PATH" ]] && OPTIONAL_ARGS+=" --record-path $RECORD_PATH"
[[ -n "$RECORD_PREFIX" ]] && OPTIONAL_ARGS+=" --record-prefix $RECORD_PREFIX"




OPTIONAL_ARGS+="$EXTRA_ARGS"

echo "Optional args: $OPTIONAL_ARGS"
# exit 0
########## Static env (unchanged) ##########


NNODES=1
GPUS_PER_NODE=4
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

CONFIG_SAVE_PATH=../tools/configs/
mkdir -p $LOG_PATH

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
CONFIG_BASENAME=$(basename ${CONFIG_NAME} .json)

########## Launch ##########
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_gpt.py \
    --flexpipe-config $CONFIG_SAVE_PATH${CONFIG_NAME} \
    --train-iters 1 \
    --eval-iters 0 \
    --lr-decay-iters 320000 \
    --vocab-file ${ROOT_PATH}/runtime/vocabs/gpt2-vocab.json \
    --merge-file ${ROOT_PATH}/runtime/vocabs/gpt2-merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 1 \
    --DDP-impl local \
    --bf16 \
    --log-path $LOG_PATH \
    --comm-shape \
    --prof-new-gpt \
    --packing \
    --system-warm 2 \
    --max-token-num 1024 \
  $OPTIONAL_ARGS \
  2>&1 | tee ${LOG_PATH_}/log_${CONFIG_BASENAME}_${CURRENT_TIME}.log