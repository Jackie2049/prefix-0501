#! /bin/bash
# source ~/anaconda3/bin/activate ~/anaconda3/envs/aceso
# source ./external/spack/share/spack/setup-env.sh
# spack load cuda@11.6

ROOT_PATH=$(pwd)
cd $ROOT_PATH/hetero_search

exp_setting=$1
op_grain=$2
search_budget=200

#### Model info ####
model_name=gpt
global_batch_size=16
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
RESULT_PATH=${ROOT_PATH}/logs-large/diff_order_recom_comm_revised_pipe_balancev1_${exp_setting}_32GPU_${min_gpus_per_stage}_${max_gpus_per_stage}_${start_num_stages}_${end_num_stages}_${min_num_pipeline}_${max_num_pipeline}_dp_${CURRENT_TIME}/${op_grain}/

save_prefix="temp"
save_prefix=diff_order_recom_comm_revised_pipe_balancev1_32GPU_${exp_setting}_${min_gpus_per_stage}_${max_gpus_per_stage}_${start_num_stages}_${end_num_stages}_${min_num_pipeline}_${max_num_pipeline}_dp_${CURRENT_TIME}


#### Search algo parameters ####
budget=$search_budget
max_num_hops=7
# init_config=/workspace/aceso/logs-large/temp_/configs/gpt/6_7B/top_configs/gpt_6_7B_temp_200_2024-11-16-6-15-15.json 
init_config=balance_v1
model_sizes=("350M" "1_3B" "2_6B" "6_7B" "13B" "70B")
num_nodes_list=(2 2 2 2 2)
gpus_per_node_list=(2 2 2 2 2)

for ((num_layer=2; num_layer<=64; num_layer=num_layer*2))
do
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
        
    nohup python3 aceso_search_dp_hete.py \
        --model-name $model_name \
        --model-size $model_size \
        --num-layer $num_layer \
        --global-batch-size $global_batch_size \
        --micro-batch-size 1 \
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
        --op-grain ${op_grain} \
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
        >/home/ymj/project/Aceso/log/motivation_exp/910Ax1x1_910B2x1x1_${model_name}_num_layer_${model_size}_gbs_${global_batch_size}_pp_mbs_1.log 1>&1
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
done
done