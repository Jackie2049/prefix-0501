

import argparse
import csv
import datetime
import json
import os 
import shutil
import time
from dataclasses import dataclass, field
from typing import List

resnet_configs = {
    "250M": ([3, 4, 6, 3], 160, 2, "fp32"),
    "500M": ([3, 4, 6, 3], 224, 2, "fp32"),
    "1B": ([3, 4, 6, 3], 320, 2, "fp32"), 
    "2B": ([3, 4, 6, 3], 448, 2, "fp32"),
    "4B": ([3, 4, 6, 3], 640, 2, "fp32"),
    "6_8B": ([3, 4, 6, 3], 320, 16, "fp32"),
    "13B": ([3, 4, 23, 3], 320, 16, "fp32"),
}

# model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (24, 2048, 1024, 1024*4, 16, 1024//16, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 2048*4, 32, 2048//32, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 2560*4, 32, 2560//32, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 4096*4, 32, 4096//32, 51200, "fp16"),
    "13B": (40, 2048, 5120, 5120*4, 40, 5120//40, 51200, "fp16"),
    "scale-layer": (1, 2048, 512, 512*4, 8, 512//8, 51200, "fp16")
}


# model_size: (num_layers, encoder_seq_length, decoder_seq_length, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
t5_configs = {
    # "220M": (12, SEQ_LEN, DECODER_SEQ_LEN, 768, 3072, 12, 64, 30592, "fp16"),
    "770M": (24, 2048, 512, 1024, 4096, 16, 64, 30592, "fp16"),
    "3B": (24, 2048, 512, 1024, 16384, 32, 128, 30592, "fp16"),
    "6B": (24, 2048, 512, 1024, 32768, 64, 128, 30592, "fp16"),
    "11B": (24, 2048, 512, 1024, 65536, 128, 128, 30592, "fp16"),    
    "22B": (48, 2048, 512, 1024, 65536, 128, 128, 30592, "fp16"),
}

## NOTE: For GPT and T5 models, we use fp16, which will introduce a "main_param" in Megatron
memory_ratio = {
    "resnet": {"main_params": 0, "optimizer": 2},
    "gpt": {"main_params": 2, "optimizer": 4},
    "t5": {"main_params": 2, "optimizer": 4},
}
def add_model_args(parser):
    group = parser.add_argument_group(title='model information')
    group.add_argument('--model-name', type=str, default=None, help='')
    group.add_argument('--model-size', type=str, default=None, help='')
    group.add_argument('--num-layers', type=int, default=None, help='')
    group.add_argument('--global-batch-size', type=int, default=None, help='')
    group.add_argument('--micro-batch-size', type=int)
    group.add_argument('--seq-len', type=int, default=2048, help='')
    group.add_argument('--decoder-seq-len', type=int, default=512, help='')
    group.add_argument('--max-tp', type=int, default=None, help='')
    group.add_argument('--num-algos', type=int, default=None, help='')
    return parser

def add_hardware_args(parser):
    group = parser.add_argument_group(title='hardware information')
    group.add_argument('--num-nodes', type=int, default=None, help='')
    group.add_argument('--num-gpus-per-node', type=int, default=None, help='')
    group.add_argument('--memory-limit', type=int, default=28000, help='')

    return parser
def add_general_args(parser):
    group = parser.add_argument_group(title='general information')
    group.add_argument('--num-ops-in-each-stage',nargs='+', type=int, default=None, help='')
    group.add_argument('--num-gpus-list', nargs='+', type=int, default=None, help='')
    group.add_argument('--resharding-stages', nargs='+', type=lambda x: x.lower() == 'true', default=None, help='')
    group.add_argument('--model-parallel-size-of-each-op', nargs='+', type=int, default=None, help='')
    group.add_argument('--data-parallel-size-of-each-op', nargs='+', type=int, default=None, help='')
    group.add_argument('--recompute-ops', nargs='+', type=int, default=None, help='')
    group.add_argument('--algo-of-each-op', nargs='+', type=int, default=None, help='')
    group.add_argument('--checkpoint-activations', nargs='+', type=lambda x: x.lower() == 'true', default=None, help='')
    group.add_argument('--max-position-embeddings', type=int, default=2048, help='')
    group.add_argument('--num-attention-heads', type=int, default=None, help='')
    group.add_argument('--hidden-size', type=int, default=None, help='')
    group.add_argument('--save-path', type=str, default="configs", help='')
    group.add_argument('--prefix', type=str, default="", help='')
    
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_hardware_args(parser)
    parser = add_general_args(parser)


    args = parser.parse_args()

    args.num_gpus = sum(args.num_gpus_list)
    print(args.num_gpus)
    # if os.path.exists(args.initial_point):
    #     with open(args.initial_point, "r") as f:
    #         config_dict = json.load(f)
    #         args.model_name = config_dict["model_name"]
    #         args.model_size = config_dict["model_size"]

    if args.model_name not in ["resnet", "gpt", "t5", "scale-layer"]:
        raise RuntimeError(f"model {args.model_name} is not supported yet.")

    if args.num_layers is None:
        if args.model_name == "resnet":
            args.num_layers = sum(resnet_configs[args.model_size][0])
        elif args.model_name == "gpt":
            args.num_layers = gpt_configs[args.model_size][0]
        elif args.model_name == "t5":
            args.num_layers = t5_configs[args.model_size][0]
        elif args.model_name == "scale-layer":
            raise RuntimeError(f"should provide --num-layers for scale-layer exp")
    if args.num_layers <= 24:
        args.print_recompute_ops = True

    if args.micro_batch_size is None:
        if args.model_name in ["resnet"]:
            args.micro_batch_size = [16, 32, 48, 64]
        else:
            args.micro_batch_size = [1, 2, 4, 8]
    if args.num_attention_heads is None:

        if args.model_name in ["gpt"]:
            args.num_attention_heads = gpt_configs[args.model_size][4]
    if args.hidden_size is None:
        if args.model_name in ["gpt"]:
            args.hidden_size = gpt_configs[args.model_size][2]
    # if args.start_num_stages is None or args.end_num_stages is None:
    #     args.start_num_stages = 1 
    #     args.end_num_stages = min(args.num_gpus, 16) #why 16?
    # if args.max_tp is None:
    #     args.max_tp = args.num_gpus_per_node

    # if args.time_budget_per_trial is None:
    #     assert args.time_budget_total is not None, "a time budget should be given, with --time-budget-total"
    #     args.time_budget_per_trial = args.time_budget_total

    # args.min_mbs = min(args.micro_batch_size)
    # if args.model_name in ["gpt", "scale-layer", "resnet"]:
    #     args.num_algos = 2
    # elif args.model_name == "t5":
    #     args.num_algos = 1

    # if args.model_name == "scale-layer":
    #     args.memory_main_params = memory_ratio["gpt"]["main_params"]
    #     args.memory_optimizer = memory_ratio["gpt"]["optimizer"]
    # else:
    #     args.memory_main_params = memory_ratio[args.model_name]["main_params"]
    #     args.memory_optimizer = memory_ratio[args.model_name]["optimizer"]

    # if args.model_name not in ["t5"]:
    #     args.resharding = True
    # else:
    #     args.resharding = False 

    cur_time = datetime.datetime.now()
    args.config_suffix = f"{cur_time.year}-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}-{cur_time.second}"
    return args

def generate_config(args):
    config = {}
    config["model_name"] = args.model_name
    config["model_size"] = args.model_size
    config["num_layers"] = args.num_layers
    config["seq_length"] = args.seq_len
    config["max_position_embeddings"] = args.max_position_embeddings
    config["num_attention_heads"] = args.num_attention_heads
    config["hidden_size"] = args.hidden_size
    config["global_batch_size"] = args.global_batch_size
    config["micro_batch_size"] = args.micro_batch_size
    config["num_stages"] = len(args.num_gpus_list)
    config["num_gpus"] = args.num_gpus_list
    config["checkpoint_activations"]=args.checkpoint_activations
    config["resharding_stages"] = args.resharding_stages
    print(args.resharding_stages)
    print(args.recompute_ops)
    config["num_ops_in_each_stage"] = args.num_ops_in_each_stage
    
    config["model_parallel_size_of_each_op"] = []
    config["data_parallel_size_of_each_op"] = []
    config["recompute_ops"] = []
    config["algo_of_each_op"] = []
    
    '''
        "model_name": "gpt",
        "model_size": "350M",
        "num_layers": 24,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "global_batch_size": 1024,
        "micro_batch_size": 8,
        "num_stages": 2,
        "num_gpus": [
            2,
            2
        ],
    '''
        
    for i in range(len(args.num_gpus_list)):
        config["model_parallel_size_of_each_op"].append([])
        config["data_parallel_size_of_each_op"].append([])
        config["recompute_ops"].append([])
        config["algo_of_each_op"].append([])
        for k in range(args.num_ops_in_each_stage[i]):
            config["model_parallel_size_of_each_op"][i].append(args.model_parallel_size_of_each_op[i])
            config["data_parallel_size_of_each_op"][i].append(args.data_parallel_size_of_each_op[i])
            config["algo_of_each_op"][i].append(args.algo_of_each_op[i])
            config["recompute_ops"][i].append(args.recompute_ops[i])



# 保存配置文件
    with open(f"{args.save_path}/{args.model_name}-{args.model_size}-{args.prefix}-{args.config_suffix}.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"config file saved to {args.save_path}/{args.model_name}-{args.model_size}-{args.prefix}-{args.config_suffix}.json")
    return config


if __name__ == "__main__":
    args = parse_args()
    generate_config(args)
'''bash_script 如下"

python generate_config.py \
--model-name gpt \
--model-size 350M \
--num-layers 24 \
--global-batch-size 32 \
--micro-batch-size 8 \
--num-ops-in-each-stage 4 \
--num-gpus-list 1 1 1 1 \
--resharding-stages False False False False \
--model-parallel-size-of-each-op 1 1 1 1 \
--data-parallel-size-of-each-op 1 1 1 1 \
--recompute-ops 0 0 0 0 \
--algo-of-each-op 0 0 0 0 \
--checkpoint-activations False False False False \
'''


