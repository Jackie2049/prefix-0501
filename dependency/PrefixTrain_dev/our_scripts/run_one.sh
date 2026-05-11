#!/bin/bash

# --- 基础环境变量设置 ---
export DEBUG_COMMUNICATE=0
export ENABLE_WEIGHT_SHARE=0
export CUDA_LAUNCH_BLOCKING=1

# --- 1. 定义实验维度 ---
# 你可以在这里注释掉不需要跑的 Agent 或方法
# AGENT_LIST=("webshop" "alfworld_4096" "search" )
AGENT_LIST=("webshop" )
METHOD_LIST=("megatron" "ours" )

# --- 2. 循环遍历组合 ---
for agent in "${AGENT_LIST[@]}"; do
    
    # 根据 Agent 匹配模型 Config
    case $agent in
        "webshop"|"alfworld_4096")
            CONFIG="dp_brief_1.5B_v1_tp2_pp2.json"
            DATA_TYPES=( "cat_think_data" "no_think_data" "data")
            ;;
        "search")
            CONFIG="dp_brief_7B_v1_tp2_pp2.json"
            DATA_TYPES=("cat_think_data" "no_think_data" "tree_data" "data")
            ;;
    esac

    for data_type in "${DATA_TYPES[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
            
            echo "========== Running: Agent=$agent, Data=$data_type, Method=$method =========="

            # 初始化方法相关的参数
            METHOD_ARGS=""
            
            # 根据方法选择参数
            if [ "$method" == "ours" ]; then
                METHOD_ARGS="--share-activation --dp-workload-balance --enable-token-level-balance"
            elif [ "$method" == "megatron" ]; then
                METHOD_ARGS=" --dp-workload-balance" # Megatron 保持默认，不加额外参数
            fi

            # 执行脚本
            # 注意：这里假设路径结构为 .../${agent}_${data_type}_data_id.pt
            bash ./our_scripts/template.sh \
                --config "$CONFIG" \
                --dp-simi 2 \
                $METHOD_ARGS \
                --input-token-path "/workspace/heteflex/tools/data/data/${agent}_${data_type}_id.pt" \
                --record-path "../log/${agent}/${data_type}/${method}/" \
                --record-prefix "${agent}_${method}_${data_type}"
            # echo "bash ./our_scripts/template.sh --config $CONFIG --dp-simi 2 $METHOD_ARGS --input-token-path /workspace/heteflex/tools/data/data/${agent}_${data_type}_data_id.pt --record-path ../log/${agent}/${data_type}/${method}/ --record-prefix ${agent}_${method}_${data_type}"
        done
    done
done