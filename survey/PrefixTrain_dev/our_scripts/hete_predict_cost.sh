ROOT_PATH=$(pwd)
cd $ROOT_PATH/hetero_search
DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-hete/
RESULT_PATH=${ROOT_PATH}/logs-large/hete_pred/
config_name=test_pred
config_path=$ROOT_PATH/logs-large/hete_pred/configs/gpt/1_3B/
# mkdir -p ${config_path}csv && mkdir mkdir -p ${config_path}top_configs
# cp single_gpu_configs/$config_name.json $ROOT_PATH/logs-large/hete_pred/configs/gpt/1_3B/top_configs/
python3 aceso_cost_model.py \
    --initial-point /workspace/aceso/logs-large/temp_/configs/gpt/6_7B/top_configs/gpt_6_7B_temp_200_2024-11-16-11-51-55.json \
    --profiled-time-path $DATABASE_PATH \
    --save-to-csv ${config_path}csv/info_$config_name.csv \
    --node-json-path ${ROOT_PATH}/hetero_search/info/node_info.json \
    --device-json-path ${ROOT_PATH}/hetero_search/info/device_info.json \
    --config-node-order-idx 2 \
    --comm-revised \
    --recom-revised
