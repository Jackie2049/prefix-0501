#! /bin/bash
# bash profile_large_gpt.sh 350M RTX3090
MASTER_ADDR=localhost
MASTER_PORT=7001
NNODES=1
NODE_RANK=0
RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-hete/

mkdir ${PROFILING_PATH}
MAX_NUM_GPUS=8
MODEL_NAME=gpt
MODEL_SIZE='350M'
GPU_NAME='910A'
mkdir ${PROFILING_PATH}/${GPU_NAME}/
mkdir ${PROFILING_PATH}/${GPU_NAME}/${MODEL_NAME}_${MODEL_SIZE}
SAVE_PROFILING_PATH=${PROFILING_PATH}/${GPU_NAME}/${MODEL_NAME}_${MODEL_SIZE}/
echo "MODEL_SIZE: $MODEL_SIZE"
for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
do
GPUS_PER_NODE=${tp_size}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

torchrun $DISTRIBUTED_ARGS \
    op_profiler.py \
    --prof-tp-size $tp_size \
    --prof-path $SAVE_PROFILING_PATH \
    --prof-cache-file ${SAVE_PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-repeat-times 40 10 \
    --prof-repeat-threshold 5000 \
    --prof-warmup-times 50 \
    --prof-warmup-threshold 100000 \
    --prof-node-rank $NODE_RANK \
    --prof-new-gpt \
    --prof-ref-data ${RUNTIME_PATH}profiled-time-eurosys/${MODEL_NAME}_op_profile.pkl \
    2>&1 | tee ${SAVE_PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log

echo [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${SAVE_PROFILING_PATH}profiling_${MODEL_NAME}.log
done
# rm -rf  /home/zzm/workspace/profiler/profiler/profiled-time-hete/910B3
for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
do
echo [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

python3 comm_profiler.py \
    --prof-path $SAVE_PROFILING_PATH \
    --prof-cache-file $SAVE_PROFILING_PATH_comm_profile.pkl \
    --prof-op-time-path $SAVE_PROFILING_PATH \
    --prof-tp-size $num_gpus \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-warmup-times 5 \
    --prof-repeat-times 20 \
    --max-data-size 40960 \
    2>&1 | tee ${SAVE_PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

echo [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${SAVE_PROFILING_PATH}profiling_${MODEL_NAME}.log

done