#! /bin/bash
# source ~/anaconda3/bin/activate ~/anaconda3/envs/aceso
# source ./external/spack/share/spack/setup-env.sh
# spack load cuda@11.6

ROOT_PATH=$(pwd)
cd $ROOT_PATH/hetero_search

exp_setting=13B
search_budget=200

#### Model info ####
model_name=gpt
global_batch_size=1024
#### Hardware info #### 
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

memory_limit=-1
min_gpus_per_stage=2
max_gpus_per_stage=8 
start_num_stages=2
end_num_stages=8
min_num_pipeline=1
max_num_pipeline=1
#### Paths ####
DATABASE_PATH=${ROOT_PATH}/profiler/npu_profile_new/
RESULT_PATH=${ROOT_PATH}/logs-large/temp_/
RESULT_PATH=${ROOT_PATH}/logs-large/norecomp_search_budget${exp_setting}_32GPU_${min_num_pipeline}_${max_num_pipeline}_dp_${CURRENT_TIME}/${op_grain}/

save_prefix="temp"
save_prefix=${exp_setting}_pipelinerange_${min_num_pipeline}_${max_num_pipeline}_dp_norecomp_search_budget${search_budget}_${CURRENT_TIME}


#### Search algo parameters ####
budget=$search_budget
max_num_hops=7
# init_config=/workspace/aceso/logs-large/temp_/configs/gpt/6_7B/top_configs/gpt_6_7B_temp_200_2024-11-16-6-15-15.json 
init_config=balance_v1
model_sizes=("350M" "1_3B" "2_6B" "6_7B" "13B" "70B")
num_nodes_list=(2 2 2 2 2)
gpus_per_node_list=(2 2 2 2 2)


for ((index=0; index<6; index=index+1))
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

    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    python3 aceso_search_dp_hete.py \
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
        --comm-revised \
        --op-grain layer \
        --save-prefix $save_prefix \
        --min-gpus-per-stage $min_gpus_per_stage \
        --max-gpus-per-stage $max_gpus_per_stage \
        --max-tp 8 \
        --enable-diff-order \
        --start-num-stages $start_num_stages \
        --end-num-stages $end_num_stages \
        --add-action-tp-dp-exchange \
        --no-add-action-tp-dp \
        --min-num-pipeline $min_num_pipeline \
        --max-num-pipeline $max_num_pipeline \
        --use-cache \
        --reduce-output \
        --no-recomp \
        2>&1 | tee ${LOG_PATH}log_${save_prefix}_${model_name}_${model_size}_budget${budget}_${memory_limit}_${CURRENT_TIME}.log
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
done