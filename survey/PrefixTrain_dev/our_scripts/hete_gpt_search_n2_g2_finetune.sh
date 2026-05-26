#! /bin/bash
# source ~/anaconda3/bin/activate ~/anaconda3/envs/aceso
# source ./external/spack/share/spack/setup-env.sh
# spack load cuda@11.6

ROOT_PATH=$(pwd)
cd $ROOT_PATH/hetero_search

exp_setting=$1
op_grain=$2
search_budget=200


#### Paths ####
DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-hete/
RESULT_PATH=${ROOT_PATH}/logs-large/temp_/
# RESULT_PATH=${ROOT_PATH}/logs-large/aceso_${exp_setting}_n2_g2_test/


#### Model info ####
model_name=gpt
global_batch_size=1024

#### Hardware info ####
memory_limit=-1

#### Search algo parameters ####
budget=$search_budget
max_num_hops=17
init_config=/workspace/aceso/logs-large/temp_/configs/gpt/6_7B/top_configs/gpt_6_7B_temp_200_2024-11-16-7-42-44.json

model_sizes=("350M" "1_3B" "2_6B" "6_7B" "13B")
num_nodes_list=(2 2 2 2 2)
gpus_per_node_list=(2 2 2 2 2)
save_prefix="temp"
# save_prefix="diff-order-${op_grain}-balance_v1_comm_revised_v3-recom_revised_v1"

for ((index=0; index<5; index=index+1))
do  
    if [ "$exp_setting" != "${model_sizes[$index]}" ]; then
        echo "Skipping ${model_sizes[$index]}"
        continue
    fi
    echo "Searching for ${model_sizes[$index]}"
    model_size=${model_sizes[$index]}

    LOG_PATH=${RESULT_PATH}hetero_search/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
    mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    python3 aceso_search.py \
        --model-name $model_name \
        --model-size $model_size \
        --global-batch-size $global_batch_size \
        --micro-batch-size 1 2 4 8 \
        --node-json-path ${ROOT_PATH}/hetero_search/info/node_info.json \
        --device-json-path ${ROOT_PATH}/hetero_search/info/device_info.json \
        --memory-limit $memory_limit \
        --log-path $LOG_PATH \
        --profiled-time-path $DATABASE_PATH \
        --config-save-path $CONFIG_SAVE_PATH \
        --config-suffix $CURRENT_TIME \
        --max-num-hops $max_num_hops \
        --time-budget-total $budget \
        --initial-point $init_config \
        --num-of-saved-configs 1 \
        --support-comm-predict \
        --enable-diff-order \
        --comm-revised \
        --recom-revised \
        --op-grain ${op_grain} \
        --config-node-order-idx 2 \
        --save-prefix $save_prefix \
        2>&1 | tee ${LOG_PATH}log_${save_prefix}_${model_name}_${model_size}_budget${budget}_${memory_limit}_${CURRENT_TIME}.log
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
    
done






