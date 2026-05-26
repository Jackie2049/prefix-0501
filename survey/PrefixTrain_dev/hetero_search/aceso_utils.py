# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import datetime
import json
import os 
import shutil
import time
from dataclasses import dataclass, field
from typing import List
from model_ops_info import get_full_op_list
import itertools
import sys
sys.path.append("./pp_simulator")
from pp_simulator.simulator import InterleavedOneFOneBGenerator, OperationExecutor, OneFOneBGenerator, EagerOneFOneBGenerator
from pp_simulator.operations import Config, HyperConfig
# model_size: (num_layers, in_channels, width_factor, params_dtype)
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
    "70B": (80 , 2048, 8192, 28672 , 64 , 8192//64, 32000, "fp16" ),

    "scale-layer": (1, 2048, 512, 512*4, 8, 512//8, 51200, "fp16")
}

llama2_config = {
    "7b": (32 ,2048, 4096, 11008, 32, 4096//32, 32000, "bf16"),
    "34b": (48 , 2048, 8192, 22016 , 64 , 8192//64, 32000, "bf16" ),
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

MAX_VALUE = 2**30
MIN_VALUE = -2**30
GLOBAL_TIMER = None 

@dataclass 
class AcesoStageInfo:
    index: int
    num_stages_behind: int
    num_gpus: int
    ops: List[str]
    recompute_ops: List[str]
    tp_size: List[int]
    dp_size: List[int]
    cp_size: List[int]
    algo: List[int]
    node_id_num: dict #key: node_id, value: node_num

@dataclass
class AcesoConfig:
    global_bs: int
    micro_bs: int
    stages: List[AcesoStageInfo]
    num_stages: int
    history: str = ""

    time_list: List[float] = field(default_factory=list)
    memory_list: List[float] = field(default_factory=list)
    compute_time_list: List[float] = field(default_factory=list)
    total_gpu_time: float = 0

    breakdown_ideal_time_per_gpu: List[float] = field(default_factory=list)
    breakdown_eff_loss_time_per_gpu: List[float] = field(default_factory=list)
    breakdown_recomp_time_per_gpu: List[float] = field(default_factory=list)    

    ## for choosing partner stages according to efficient time
    efficient_time_list: List[float] = field(default_factory=list)
    ## used for adaptive model
    adaptive_times: int = 0
    
    each_stage_time_breakdown: List[List[float]] = field(default_factory=list)
    each_stage_memory_breakdown: List[List[float]] = field(default_factory=list)


@dataclass
class HybridConfig:

    num_pipelines: int 
    pipelines: List[AcesoConfig]  = field(default_factory=list)
    global_bs_pipline: List[int] = field(default_factory=list)
    micro_bs_pipline: List[int] = field(default_factory=list)
    num_mb_pipline: List[int] = field(default_factory=list)
    time_list_pipline : List[float] = field(default_factory=list)

def debug_info(info, print_debug_info):
    if print_debug_info:
        print(info)

def get_config(num_ops_list, tp_size_list, dp_size_list,cp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num):
    op_start_index = 0
    num_stages = len(num_ops_list)
    stages_info_list = []
    for i in range(num_stages):
        stage_info = AcesoStageInfo(
            index = i, 
            num_stages_behind = (num_stages - 1 - i),
            num_gpus = tp_size_list[op_start_index] * dp_size_list[op_start_index],
            ops = list(full_op_list[op_start_index: op_start_index + num_ops_list[i]]),
            recompute_ops = list(recompute_ops[op_start_index: op_start_index + num_ops_list[i]]),
            tp_size = list(tp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            dp_size = list(dp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            cp_size = list(cp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            algo = list(algo_list[op_start_index: op_start_index + num_ops_list[i]]),
            node_id_num=node_id_num[i]
            )
        stages_info_list.append(stage_info)
        op_start_index += num_ops_list[i]

    current_config = AcesoConfig(global_bs=global_batch_size, micro_bs=aggregate_mbs, stages=stages_info_list, num_stages=num_stages)
    return current_config

def config_details(config, get_string=False):
    if config is None:
        return ""
    num_ops_stage = []
    tp_size_list = []
    dp_size_list = []
    recompute_ops = []
    algo_list = []
    base_batch_size = config.micro_bs
    for i in range(config.num_stages):
        num_ops_stage.append(len(config.stages[i].ops))
        tp_size_list.append(config.stages[i].tp_size)
        dp_size_list.append(config.stages[i].dp_size)
        recompute_ops.append(config.stages[i].recompute_ops)
        algo_list.append(config.stages[i].algo)
    if get_string:
        return f"{num_ops_stage}, {tp_size_list}, {dp_size_list}, {recompute_ops}, {base_batch_size}, {algo_list}"
    else:
        return num_ops_stage, tp_size_list, dp_size_list, recompute_ops, base_batch_size, algo_list

def dump_config_to_json(config, file_name, args):
    if args.model_name == "scale-layer":
        model_name = "gpt"
        model_size = "scale-layer"
    else:
        model_name = args.model_name
        model_size = args.model_size
    num_layers = args.num_layers
    config_dict = {}
    config_dict["node_info"] = config.node_info
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size
    if model_name == "resnet":
        num_layers, in_channels, width_factor, _ = resnet_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["in_channels"] = in_channels
        config_dict["width_factor"] = width_factor
    elif model_name == "gpt":
        _, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["seq_length"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size        
    elif model_name == "t5":
        _, encoder_seq_length, decoder_seq_length, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = t5_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["encoder_seq_length"] = encoder_seq_length
        config_dict["decoder_seq_length"] = decoder_seq_length
        config_dict["max_position_embeddings"] = encoder_seq_length
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["kv_channels"] = kv_channels
        config_dict["hidden_size"] = hidden_size
        config_dict["ffn_hidden_size"] = ffn_hidden_size        
    else:
        raise RuntimeError(f"{model_name} not supportted.")

    config_dict["global_batch_size"] = config.global_bs
    config_dict["micro_batch_size"] = config.micro_bs
    config_dict["num_stages"] = config.num_stages

    tp_size_of_each_op = []
    dp_size_of_each_op = []
    recompute_ops = []
    algo_of_each_op = []
    num_ops_in_each_stage = []
    config_dict["num_gpus"] = []
    config_dict["checkpoint_activations"] = []
    config_dict["resharding_stages"] = []
    for i in range(config.num_stages):
        tp_size_of_each_op.append(config.stages[i].tp_size)
        dp_size_of_each_op.append(config.stages[i].dp_size)
        recompute_ops.append(config.stages[i].recompute_ops)
        algo_of_each_op.append(config.stages[i].algo)
        num_ops_in_each_stage.append(len(config.stages[i].ops))

        config_dict["num_gpus"].append(config.stages[i].num_gpus)
        if max(config.stages[i].recompute_ops) > 0:
            config_dict["checkpoint_activations"].append(True)
        else:
            config_dict["checkpoint_activations"].append(False)
        if max(config.stages[i].tp_size) != min(config.stages[i].tp_size) \
            or max(config.stages[i].dp_size) != min(config.stages[i].dp_size) \
            or max(config.stages[i].algo) != min(config.stages[i].algo):
            config_dict["resharding_stages"].append(True)
        else:
            config_dict["resharding_stages"].append(False)

    config_dict["num_ops_in_each_stage"] = num_ops_in_each_stage
    config_dict["model_parallel_size_of_each_op"] = tp_size_of_each_op
    config_dict["data_parallel_size_of_each_op"] = dp_size_of_each_op
    config_dict["recompute_ops"] = recompute_ops
    config_dict["algo_of_each_op"] = algo_of_each_op
    json.dump(config_dict, open(file_name, 'w'), indent=4)
    print(f"config has been saved to {file_name}")    


def dump_hybrid_config_to_json(config: HybridConfig, file_name, args):


    if args.model_name == "scale-layer":
        model_name = "gpt"
        model_size = "scale-layer"
    else:
        model_name = args.model_name
        model_size = args.model_size
    num_layers = args.num_layers
    config_dict = {}
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size
    if model_name == "resnet":
        num_layers, in_channels, width_factor, _ = resnet_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["in_channels"] = in_channels
        config_dict["width_factor"] = width_factor
    elif model_name == "gpt":
        _, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["seq_length"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size        
    elif model_name == "t5":
        _, encoder_seq_length, decoder_seq_length, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = t5_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["encoder_seq_length"] = encoder_seq_length
        config_dict["decoder_seq_length"] = decoder_seq_length
        config_dict["max_position_embeddings"] = encoder_seq_length
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["kv_channels"] = kv_channels
        config_dict["hidden_size"] = hidden_size
        config_dict["ffn_hidden_size"] = ffn_hidden_size        
    else:
        raise RuntimeError(f"{model_name} not supportted.")

    config_dict["pipe"]= []
    for pipe_idx in range(config.num_pipelines):
        config_dict_pipe = {}
        config_dict_pipe["global_batch_size"] = config.pipelines[pipe_idx].global_bs
        config_dict_pipe["micro_batch_size"] = config.pipelines[pipe_idx].micro_bs
        config_dict_pipe["num_stages"] = config.pipelines[pipe_idx].num_stages
        config_dict_pipe["node_info"] = config.pipelines[pipe_idx].node_info

    

        tp_size_of_each_op = []
        dp_size_of_each_op = []
        cp_size_of_each_op = []
        tp_size_of_each_stage = []
        dp_size_of_each_stage = []
        cp_size_of_each_stage = []
        recompute_ops = []
        algo_of_each_op = []
        algo_of_each_stage = []
        num_ops_in_each_stage = []
        config_dict_pipe["num_gpus"] = []
        config_dict_pipe["checkpoint_activations"] = []
        config_dict_pipe["resharding_stages"] = []

        for i in range(config.pipelines[pipe_idx].num_stages):
            tp_size_of_each_op.append(config.pipelines[pipe_idx].stages[i].tp_size)
            dp_size_of_each_op.append(config.pipelines[pipe_idx].stages[i].dp_size)
            cp_size_of_each_op.append(config.pipelines[pipe_idx].stages[i].cp_size)
            tp_size_of_each_stage.append(config.pipelines[pipe_idx].stages[i].tp_size[0])
            dp_size_of_each_stage.append(config.pipelines[pipe_idx].stages[i].dp_size[0])
            cp_size_of_each_stage.append(config.pipelines[pipe_idx].stages[i].cp_size[0])
            recompute_ops.append(config.pipelines[pipe_idx].stages[i].recompute_ops)
            algo_of_each_op.append(config.pipelines[pipe_idx].stages[i].algo)
            algo_of_each_stage.append(config.pipelines[pipe_idx].stages[i].algo[0])
            num_ops_in_each_stage.append(len(config.pipelines[pipe_idx].stages[i].ops))

            config_dict_pipe["num_gpus"].append(config.pipelines[pipe_idx].stages[i].num_gpus)
            if max(config.pipelines[pipe_idx].stages[i].recompute_ops) > 0:
                config_dict_pipe["checkpoint_activations"].append(True)
            else:
                config_dict_pipe["checkpoint_activations"].append(False)
            if max(config.pipelines[pipe_idx].stages[i].tp_size) != min(config.pipelines[pipe_idx].stages[i].tp_size) \
                or max(config.pipelines[pipe_idx].stages[i].dp_size) != min(config.pipelines[pipe_idx].stages[i].dp_size) \
                or max(config.pipelines[pipe_idx].stages[i].cp_size) != min(config.pipelines[pipe_idx].stages[i].cp_size) \
                or max(config.pipelines[pipe_idx].stages[i].algo) != min(config.pipelines[pipe_idx].stages[i].algo):
                config_dict_pipe["resharding_stages"].append(True)
            else:
                config_dict_pipe["resharding_stages"].append(False)

        config_dict_pipe["num_ops_in_each_stage"] = num_ops_in_each_stage
        if(args.save_brief_format):
            config_dict_pipe["model_parallel_size_of_each_stage"] = tp_size_of_each_stage
            config_dict_pipe["data_parallel_size_of_each_stage"] = dp_size_of_each_stage
            config_dict_pipe["context_parallel_size_of_each_stage"] = cp_size_of_each_stage
            config_dict_pipe["algo_of_each_stage"] = algo_of_each_stage
        else:
            config_dict_pipe["model_parallel_size_of_each_op"] = tp_size_of_each_op
            config_dict_pipe["data_parallel_size_of_each_op"] = dp_size_of_each_op
            config_dict_pipe["context_parallel_size_of_each_op"] = cp_size_of_each_op
            config_dict_pipe["algo_of_each_op"] = algo_of_each_op

        config_dict_pipe["recompute_ops"] = recompute_ops
        config_dict["pipe"].append(config_dict_pipe)

    json.dump(config_dict, open(file_name, 'w'), indent=4)
    print(f"config has been saved to {file_name}")    


def read_config_from_json(args, return_config_dict=False):
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)

    model_name = config_dict["model_name"]
    num_layers = config_dict["num_layers"]
    model_size = config_dict["model_size"]

    aggregate_mbs = config_dict["micro_batch_size"]
    global_batch_size = config_dict["global_batch_size"]
    num_ops_list = config_dict["num_ops_in_each_stage"]
    tp_size_list = []
    if(config_dict.get("model_parallel_size_of_each_op")!=None):
        for _tp_size_list in config_dict["model_parallel_size_of_each_op"]:
            tp_size_list += _tp_size_list
    else:
        assert "model_parallel_size_of_each_stage" in config_dict
        for stage_idx in range(len(config_dict["model_parallel_size_of_each_stage"])) :
            tp_size_list += [config_dict["model_parallel_size_of_each_stage"][stage_idx]] * num_ops_list[stage_idx]

    dp_size_list = []
    if(config_dict.get("data_parallel_size_of_each_op")!=None):
        for _dp_size_list in config_dict["data_parallel_size_of_each_op"]:
            dp_size_list += _dp_size_list        
    else:
        assert "data_parallel_size_of_each_stage" in config_dict
        for stage_dix in range(len(config_dict["data_parallel_size_of_each_stage"])):
            dp_size_list += [config_dict["data_parallel_size_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]
    
    cp_size_list = []
    if(config_dict.get("context_parallel_size_of_each_op")!=None):
        for _cp_size_list in config_dict["context_parallel_size_of_each_op"]:
            cp_size_list += _cp_size_list        
    else:
        assert "context_parallel_size_of_each_stage" in config_dict
        for stage_dix in range(len(config_dict["context_parallel_size_of_each_stage"])):
            cp_size_list += [config_dict["context_parallel_size_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]

    recompute_ops = []
    for _recompute_ops in config_dict["recompute_ops"]:
        recompute_ops += _recompute_ops

    algo_list = []
    if(config_dict.get("algo_of_each_op")!=None):
        for _algo_list in config_dict["algo_of_each_op"]:
            algo_list += _algo_list        
    else:
        assert "algo_of_each_stage" in config_dict
        for stage_dix in range(len(config_dict["algo_of_each_stage"])):
            algo_list += [config_dict["algo_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]
     
    full_op_list = get_full_op_list(args)

    num_gpus_list = config_dict["num_gpus"]
    
    # args.node_order = args.gpu_type_num_dict
    args.num_gpus_per_node_list=[]
    args.gpu_type_list = []
    for key in args.node_order.keys():
        args.num_gpus_per_node_list.append(args.node_order[key]["GPU_NUM"])
        args.gpu_type_list.append(args.node_order[key]["GPU"])
    if(config_dict.get("node_info")!=None):
        args.node_order = config_dict["node_info"]
    
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    
    # print(f"node_id_num: {node_id_num}")
    if return_config_dict:
        return get_config(num_ops_list, tp_size_list, dp_size_list, cp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num), config_dict
    else:
        return get_config(num_ops_list, tp_size_list, dp_size_list, cp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num)

def read_hybrid_config_from_json(args):
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)

    
    model_name = config_dict["model_name"]
    num_layers = config_dict["num_layers"]
    model_size = config_dict["model_size"]
    num_pipelines = len (config_dict["pipe"])
    hybridconfig = HybridConfig(num_pipelines=num_pipelines)

    args.num_gpus_per_node_list_pipe=[]
    args.gpu_type_list_pipe = []
    for pipe_idx in range(num_pipelines):
        aggregate_mbs = config_dict["pipe"][pipe_idx]["micro_batch_size"]
        global_batch_size = config_dict["pipe"][pipe_idx]["global_batch_size"]
        num_ops_list = config_dict["pipe"][pipe_idx]["num_ops_in_each_stage"]

        tp_size_list = []
        if(config_dict["pipe"][pipe_idx].get("model_parallel_size_of_each_op")!=None):
            for _tp_size_list in config_dict["pipe"][pipe_idx]["model_parallel_size_of_each_op"]:
                tp_size_list += _tp_size_list
        else:
            assert "model_parallel_size_of_each_stage" in config_dict["pipe"][pipe_idx]
            for stage_idx in range(len(config_dict["pipe"][pipe_idx]["model_parallel_size_of_each_stage"])) :
                tp_size_list += [config_dict["pipe"][pipe_idx]["model_parallel_size_of_each_stage"][stage_idx]] * num_ops_list[stage_idx]

        dp_size_list = []
        if(config_dict["pipe"][pipe_idx].get("data_parallel_size_of_each_op")!=None):
            for _dp_size_list in config_dict["pipe"][pipe_idx]["data_parallel_size_of_each_op"]:
                dp_size_list += _dp_size_list        
        else:
            assert "data_parallel_size_of_each_stage" in config_dict["pipe"][pipe_idx]
            for stage_dix in range(len(config_dict["pipe"][pipe_idx]["data_parallel_size_of_each_stage"])):
                dp_size_list += [config_dict["pipe"][pipe_idx]["data_parallel_size_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]
        
        cp_size_list = []
        if(config_dict["pipe"][pipe_idx].get("context_parallel_size_of_each_op")!=None):
            for _cp_size_list in config_dict["pipe"][pipe_idx]["context_parallel_size_of_each_op"]:
                cp_size_list += _cp_size_list        
        else:
            assert "context_parallel_size_of_each_stage" in config_dict["pipe"][pipe_idx]
            for stage_dix in range(len(config_dict["pipe"][pipe_idx]["context_parallel_size_of_each_stage"])):
                cp_size_list += [config_dict["pipe"][pipe_idx]["context_parallel_size_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]

        recompute_ops = []
        for _recompute_ops in config_dict["pipe"][pipe_idx]["recompute_ops"]:
            recompute_ops += _recompute_ops

        algo_list = []
        if(config_dict["pipe"][pipe_idx].get("algo_of_each_op")!=None):
            for _algo_list in config_dict["pipe"][pipe_idx]["algo_of_each_op"]:
                algo_list += _algo_list        
        else:
            assert "algo_of_each_stage" in config_dict["pipe"][pipe_idx]
            for stage_dix in range(len(config_dict["pipe"][pipe_idx]["algo_of_each_stage"])):
                algo_list += [config_dict["pipe"][pipe_idx]["algo_of_each_stage"][stage_dix]] * num_ops_list[stage_dix]

        full_op_list = get_full_op_list(args)

        num_gpus_list = config_dict["pipe"][pipe_idx]["num_gpus"]
        
        # args.node_order = args.gpu_type_num_dict
        args.num_gpus_per_node_list=[]
        args.gpu_type_list = []
        for key in args.node_order[pipe_idx].keys():
            args.num_gpus_per_node_list.append(args.node_order[pipe_idx][key]["GPU_NUM"])
            args.gpu_type_list.append(args.node_order[pipe_idx][key]["GPU"])
        args.num_gpus_per_node_list_pipe.append(args.num_gpus_per_node_list)
        args.gpu_type_list_pipe.append(args.gpu_type_list)
        if(config_dict.get("node_info")!=None):
            args.node_order = config_dict["node_info"]
        
        node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
        
        hybridconfig.pipelines.append(get_config(num_ops_list, tp_size_list, dp_size_list, cp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num))
        hybridconfig.global_bs_pipline.append(global_batch_size)
        hybridconfig.micro_bs_pipline.append(aggregate_mbs)
        hybridconfig.num_mb_pipline.append(global_batch_size//aggregate_mbs)

    return hybridconfig

def save_config_info_to_csv(config, reserved_mem_list, file_name):
    info_to_csv = [["stage-index", "time", "memory(total)", "memory(normal)", "memory(reserved)"]]
    for i in range(config.num_stages):
        info_to_csv.append([f"stage-{i}", f"{config.time_list[i]:.2f}", f"{config.memory_list[i]:.2f}", f"{(config.memory_list[i]-reserved_mem_list[i]):.2f}", f"{reserved_mem_list[i]:.2f}"])
    
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)

def save_distribution_to_csv(num_targets_list, num_hops_list, file_name):
    info_to_csv = [["num_targets"] + num_targets_list, ["num_hops"] + num_hops_list]

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)

class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.start_time = time.time()
        self.elapsed_history = 0.0
        self.elapsed_list = []

    def start(self):
        """Start the timer."""
        assert not self.started_, f"{self.name_} timer has already been started"
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        elapsed_time = time.time() - self.start_time
        self.elapsed_ += (elapsed_time)
        self.elapsed_list.append(elapsed_time)
        self.started_ = False     
        self.count += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.elapsed_history = (time.time() - self.start_time)

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
            self.count = 0
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_

    def elapsed_since_first_invoke(self, reset=True):
        """Calculate the elapsed time."""
        return time.time() - self.start_time


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, reset=True):
        print(f"===== timers ====")
        for name in sorted(self.timers):
            count = self.timers[name].count
            if count == 0:
                elapsed_time = self.timers[name].elapsed_history
            else:
                elapsed_time = self.timers[name].elapsed(reset=reset)
            print('{}: {:.2f} s [count = {}]'.format(name, elapsed_time, count),flush=True)
        print(f"===== end of timers ====\n",flush=True)
def print_args(args):
    """Print arguments."""
    print('------------------------ arguments ------------------------',
            flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------',
            flush=True)

def add_model_args(parser):
    group = parser.add_argument_group(title='model information')
    group.add_argument('--model-name', type=str, default=None, help='')
    group.add_argument('--model-size', type=str, default=None, help='')
    group.add_argument('--num-layers', type=int, default=None, help='')
    group.add_argument('--global-batch-size', type=int, default=None, help='')
    group.add_argument('--micro-batch-size', nargs='+', type=int, default=None, help='')
    group.add_argument('--seq-len', type=int, default=2048, help='')
    group.add_argument('--decoder-seq-len', type=int, default=512, help='')
    group.add_argument('--max-tp', type=int, default=None, help='')
    group.add_argument('--max-cp', type=int, default=None, help='')
    group.add_argument('--num-algos', type=int, default=None, help='')
    group.add_argument('--num-ops-each-layer', type=int, default=13, help='')

    return parser

def add_hardware_args(parser):
    group = parser.add_argument_group(title='hardware information')
    group.add_argument('--num-nodes', type=int, default=None, help='')
    group.add_argument('--num-gpus-per-node', type=int, default=None, help='')
    group.add_argument('--memory-limit', type=int, default=28000, help='')

    return parser

def add_path_args(parser):
    group = parser.add_argument_group(title='path information')
    group.add_argument('--log-path', type=str, default=None, help='')
    group.add_argument('--profiled-time-path', type=str, default=None, help='')
    group.add_argument('--config-save-path', type=str, default=None, help='')
    group.add_argument('--config-suffix', type=str, default=None, help='')
    group.add_argument('--save-prefix', type=str, default="", help='')

    return parser

def add_budget_args(parser):
    group = parser.add_argument_group(title='budgets for the search algorithm')
    group.add_argument('--max-num-hops', type=int, default=None, help='')
    group.add_argument('--max-num-trials', type=int, default=100, help='max number of search time') 
    group.add_argument('--time-budget-per-trial', type=int, default=None, help='')
    group.add_argument('--time-budget-total', type=int, default=10, help='Total time budget for the search')
    group.add_argument('--start-num-stages', type=int, default=None, help='')
    group.add_argument('--end-num-stages', type=int, default=None, help='')
    group.add_argument('--op-grain', type=str ,default="operator",choices=["operator", "block","layer","other"], help='')
    group.add_argument('--bisection', action='store_true', help='bisect the dec_op operator')

    return parser

def add_heuristic_args(parser):
    group = parser.add_argument_group(title='heuristics in the search algorithm')
    group.add_argument('--op-group-size', type=int, default=1, help='')
    group.add_argument('--max-op-move-steps', type=int, default=5, help='')
    group.add_argument('--memory-pred-type', type=str, default='MAX', help='')
    group.add_argument('--check-recompute-with-group', action='store_true', help='')
    group.add_argument('--initial-point', type=str, default="balance", help='')
    group.add_argument('--high-memory-rate', type=float, default=0.9, help='')

    return parser

def add_debug_args(parser):
    group = parser.add_argument_group(title='debug arguments')
    group.add_argument('--print-debug-info', action='store_true', help='')
    group.add_argument('--print-move-op-details', action='store_true', help='')
    group.add_argument('--print-recompute-ops', action='store_true', help='')
    group.add_argument('--print-recomp-debug-info', action='store_true', help='')
    
    return parser

def add_test_args(parser):
    group = parser.add_argument_group(title='arguments under development')
    group.add_argument('--do-not-consider-shared-tensor-space', action='store_false', help='', dest='consider_shared_space')
    group.add_argument('--do-not-consider-reserved-space', action='store_false', help='', dest='consider_reserved_space')
    group.add_argument('--predict-delta-time', action='store_true', help='')
    group.add_argument('--do-not-use-flex-recompute', action='store_false', help='', dest='flex_recompute')
    group.add_argument('--add-action-finetune-dim', action='store_true', help='')
    group.add_argument('--add-action-finetune-algo', action='store_true', help='')
    group.add_argument('--add-action-tp-dp-exchange', action='store_true', help='')
    group.add_argument('--peak-mem-in-backward', type=int, default=0, help='')
    group.add_argument('--add-action-tune-tp-dp', action='store_true', help='')
    group.add_argument('--finetune-after-trial', type=int, default=0, help='')
    group.add_argument('--no-multi-process', action='store_false', help='', dest='multi_process')
    group.add_argument('--random-order-actions', action='store_true', help='')
    group.add_argument('--support-comm-predict', action='store_true', help='')
    group.add_argument('--forbid-turn-back', action='store_true', help='')
    group.add_argument('--sort-metric', type=str, default='max_stage_time', help='')
    group.add_argument('--print-gpu-mig-details', action='store_true', help='')
    group.add_argument('--finetune-tp-dp-after-trial', action='store_true', help='')
    group.add_argument('--init-dim', type=str, default="tp", help='')
    group.add_argument('--num-partners-in-op-mig', type=int, default=1, help='')
    group.add_argument('--do-not-continue-when-fail', action='store_false', help='', dest='continue_when_fail')
    group.add_argument('--adaptive-hyper-parameters', type=int, default=5, help='')
    group.add_argument('--num-of-saved-configs', type=int, default=1, help='')
    group.add_argument('--simple-prim-mbs', action='store_true', help='')
    group.add_argument('--simple-prim-mig', action='store_true', help='')
    group.add_argument('--only-top-1-target', action='store_true', help='')
    group.add_argument('--consider-collective-memory', action='store_true', help='')
    group.add_argument('--save-to-csv', type=str, default=None, help='')
    group.add_argument('--statistic-search-time', action='store_true', help='')
    group.add_argument('--save-brief-format', action='store_true', help='')
    
    return parser

def add_hetero_search_args(parser):
    group = parser.add_argument_group(title='hetero search arguments')
    # group.add_argument('--num-gpus-per-node-list', nargs='+', type=int, default=None, help='')
    # group.add_argument('--gpu-type-list', nargs='+', type=str, default=None, help='')
    group.add_argument('--node-json-path', type=str, default=None, help='Actual node information, like gpu_type and gpu_num')
    group.add_argument('--node-info', type=json.loads, default=None, help='Actual node information, like gpu_type and gpu_num')
    group.add_argument('--enable-diff-order', action='store_true', help='enable different order of nodes')
    group.add_argument('--device-json-path', type=str, default=None, help='GPU information, like memory')
    group.add_argument('--comm-revised', action='store_true', help='enable revised communication model')
    group.add_argument('--recom-revised', action='store_true', help='enable revised recom model')
    group.add_argument('--config-node-order-idx', type = int, default=None, help='node order index')
    group.add_argument('--end-dg-idx ', type=int, default=2e32, help='')
    group.add_argument('--start-dg-idx', type=int, default=-1, help='')
    group.add_argument('--min-gpus-per-stage', type=int, default=1, help='')
    group.add_argument('--max-gpus-per-stage', type=int, default=640, help='')
    group.add_argument('--num-multi-process', type=int, default=32, help='')
    group.add_argument('--num-of-show-configs', type=int, default=5, help='')
    group.add_argument('--no-add-action-tp-dp', action='store_false', help='', dest='add_action_tp_dp')
    group.add_argument('--min-num-pipeline', type=int, default=1, help='')
    group.add_argument('--max-num-pipeline', type=int, default=1, help='')
    group.add_argument('--reduce-save-result', action = 'store_true')
    group.add_argument('--test-search-time', action = 'store_true')
    group.add_argument('--reduce-output', action = 'store_true')
    group.add_argument('--easy-mode', action = 'store_true')
    group.add_argument('--use-cache', action = 'store_true')
    group.add_argument('--optimal-prune', action = 'store_true')
    group.add_argument('--upper-bound-rate', type=float, default=1.5, help='')
    group.add_argument('--sort-pipeline', action = 'store_true')
    group.add_argument('--dp-find-pipeline', action = 'store_true')
    group.add_argument('--homo-stage', action = 'store_true')
    group.add_argument('--elastic-time', action = 'store_true')
    group.add_argument('--add-action-tp-cp-dp-exchange', action = 'store_true')

    return parser

def add_exp_args(parser):
    group = parser.add_argument_group(title='experiment arguments')
    group.add_argument('--no-recomp', action = 'store_true')
    group.add_argument('--no-dp', action = 'store_true')
    return parser
global_args = None

def check_initial_point(args):
    with open(args.initial_point, "r") as f:
        config_dict = json.load(f)
        model_name = config_dict["model_name"]
        model_size = config_dict["model_size"]
        num_layers = config_dict["num_layers"]

        if(args.model_name != None):
            assert args.model_name == model_name, f"model name in the initial point is {model_name}, not {args.model_name}"
        else:
            args.model_name = model_name
        if(args.model_size != None):
            assert args.model_size == model_size, f"model size in the initial point is {model_size}, not {args.model_size}"
        else:
            args.model_size = model_size
        if(args.num_layers != None):
            assert args.num_layers == num_layers, f"num_layers in the initial point is {num_layers}, not {args.num_layers}"
        else:
            args.num_layers = num_layers
        if(config_dict.get("pipe")==None):
            global_bs = config_dict["global_batch_size"]
            if(args.global_batch_size != None):
                assert args.global_batch_size == global_bs, f"global batch size in the initial point is {global_bs}, not {args.global_batch_size}"
            else:
                args.global_batch_size = global_bs
            args.num_gpus = sum(config_dict["num_gpus"])

        # if(args.num_nodes != None and args.num_gpus_per_node_list != None): #TODO
        #     assert args.num_nodes == len(args.num_gpus_per_node_list), f"num_nodes in the initial point is {len(args.num_gpus_per_node_list)}, not {args.num_nodes}"
   

def parse_args():
    global global_args
    if global_args is not None:
        return global_args
    # else:
    #     print("global_args is None, parsing args again")
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_hardware_args(parser)
    parser = add_path_args(parser)
    parser = add_budget_args(parser)
    parser = add_heuristic_args(parser)
    parser = add_debug_args(parser)
    parser = add_test_args(parser)
    parser = add_hetero_search_args(parser)
    parser = add_exp_args(parser)
    args = parser.parse_args()

    if args.initial_point is not None and os.path.exists(args.initial_point):
        check_initial_point(args)
    else:
        pass
        # print(f"initial point {args.initial_point} does not exist.")
    # args.num_gpus = args.num_gpus_per_node * args.num_nodes


    config_dict = {}
    if os.path.exists(args.initial_point):
        with open(args.initial_point, "r") as f:
            config_dict = json.load(f)
            args.model_name = config_dict["model_name"]
            args.model_size = config_dict["model_size"]
            args.seq_len = config_dict.get("seq_length", 2048)
    if(config_dict.get("pipe")!=None):
            args.gpu_type_num_dict = config_dict["pipe"][0]["node_info"]
            args.max_tp = 8
    else:
        node_id = 0
        if(config_dict.get("node_info") !=None):
            gpu_type_num_dict = config_dict["node_info"]
            args.gpu_type_num_dict = {}
            for key in gpu_type_num_dict.keys():
                if(gpu_type_num_dict[key].get("NODE_NUM")!=None):
                    for i in range (gpu_type_num_dict[key]["NODE_NUM"]):
                        args.gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict[key]["GPU"], "GPU_NUM": gpu_type_num_dict[key]["GPU_NUM"]}
                        node_id +=1  
                else:
                    args.gpu_type_num_dict = gpu_type_num_dict
        else:
            if(args.node_info == None):
                assert args.node_json_path is not None  , "node_json_path should be provided"
                with open(args.node_json_path, "r") as f:
                    gpu_type_num_dict_ = json.load(f)
            else:
                gpu_type_num_dict_ = args.node_info
            node_id = 0
            args.gpu_type_num_dict = {}
            for key in gpu_type_num_dict_.keys():
                for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
                    args.gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
                    node_id +=1    
        args.num_nodes = len(args.gpu_type_num_dict)
        args.max_num_pipeline = min(args.max_num_pipeline,args.num_nodes)
        args.num_gpus = sum([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
        args.gpu_type_set = [args.gpu_type_num_dict[f"{i}"]["GPU"] for i in range(args.num_nodes)]
        args.gpu_type_set = list(set(args.gpu_type_set))
        if args.max_tp is None:
            # args.max_tp = args.num_gpus_per_node
            args.max_tp = max([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)]) #TODO different nodes may have different max_tp
        # print("args.max_tp",args.max_tp)
        if args.start_num_stages is None or args.end_num_stages is None:
            args.start_num_stages = 1 
            args.end_num_stages = min(args.num_gpus, 16) #why 16?

    if args.max_cp is None:
        args.max_cp = args.max_tp
    if args.device_json_path is not None:
        with open(args.device_json_path, "r") as f:
            device_json = json.load(f)
            args.device_info = device_json
    # print(args.device_info)
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
    # if args.num_layers <= 24:
    #     args.print_recompute_ops = True

    if args.micro_batch_size is None:
        if args.model_name in ["resnet"]:
            args.micro_batch_size = [16, 32, 48, 64]
        else:
            args.micro_batch_size = [1, 2, 4, 8]


    if args.time_budget_per_trial is None   :
        assert args.time_budget_total is not None, "a time budget should be given, with --time-budget-total"
        args.time_budget_per_trial = args.time_budget_total

    args.min_mbs = min(args.micro_batch_size)
    if args.model_name in ["gpt", "scale-layer", "resnet"]:
        args.num_algos = 2
    elif args.model_name == "t5":
        args.num_algos = 1

    if args.model_name == "scale-layer":
        args.memory_main_params = memory_ratio["gpt"]["main_params"]
        args.memory_optimizer = memory_ratio["gpt"]["optimizer"]
    else:
        args.memory_main_params = memory_ratio[args.model_name]["main_params"]
        args.memory_optimizer = memory_ratio[args.model_name]["optimizer"]

    if args.model_name not in ["t5"]:
        args.resharding = True
    else:
        args.resharding = False 


    cur_time = datetime.datetime.now()
    args.config_suffix = f"{cur_time.year}-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}-{cur_time.second}"
    global_args = args 
    return args


def update_args(new_args):
    global global_args
    global_args = new_args 


def generate_balance_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # 如果当前stage的gpu数量小于最大gpu数量，且当前stage的gpu数量小于剩余gpu数量
            # 则将当前stage的gpu数量翻倍
            if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): #TODO use args.max_tp
            # if sum(num_gpus_list) < num_gpus  and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): 
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        print(f"sum(num_gpus_list) != num_gpus,{sum(num_gpus_list)} != {num_gpus}")
        return None

    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    
        


    num_ops = len(full_op_list)
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]            
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)

    return initial_config


def generate_balance_config_v1(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # 如果当前stage的gpu数量小于最大gpu数量，且当前stage的gpu数量小于剩余gpu数量
            # 则将当前stage的gpu数量翻倍
            # if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): #TODO use args.max_tp
            if sum(num_gpus_list) < num_gpus  and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): 
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        # print(f"sum(num_gpus_list) != num_gpus,{sum(num_gpus_list)} != {num_gpus}")
        min_stage = num_gpus_list.index(min(num_gpus_list))
        num_gpus_list[min_stage] += num_gpus - sum(num_gpus_list)
        # return 

    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    
        


    num_ops = len(full_op_list)
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]            
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)

    return initial_config

   
def generate_test_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 

    num_gpus_list = [1 for _ in range(num_stages)]
    num_gpus_left = num_gpus - sum(num_gpus_list)
    while num_gpus_left > 0:
        initial_num_gpus_left = num_gpus_left
        for i in range(num_stages):
            if num_gpus_list[num_stages - 1 - i] <= num_gpus_left and num_gpus_list[num_stages - 1 - i] == min(num_gpus_list) and num_gpus_list[num_stages - 1 - i] <= 4:
                num_gpus_left -= num_gpus_list[num_stages - 1 - i]
                num_gpus_list[num_stages - 1 - i] *= 2 
                break 
        if num_gpus_left == initial_num_gpus_left:
            break
    if sum(num_gpus_list) != num_gpus:
        return generate_initial_config(full_op_list, num_stages, num_gpus)
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    num_ops = len(full_op_list)
    num_ops_per_stage = []
    for i in range(num_stages):
        num_ops_per_stage.append(int(num_ops * (num_gpus_list[i]/num_gpus)))
    num_ops_per_stage[-1] += num_ops - sum(num_ops_per_stage)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
        dp_size_list += [1 for _ in range(num_ops_per_stage[i])]

    recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

def generate_imbalance_gpu_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 
    num_ops = len(full_op_list)
    recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]    
    ## op distribution
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    ## gpu distribution
    num_gpus_list = [1 for _ in range(num_stages)]
    num_gpus_remained = num_gpus - sum(num_gpus_list)
    print(f"{num_gpus_list}")
    micro_bs_index = 0
    while num_gpus_remained > 0:
        found = False
        for i in range(num_stages):
            # if num_gpus_list[num_stages - 1 - i] <= num_gpus_remained and \
            #     (num_gpus_list[num_stages - 1 - i] * 2 // args.max_tp == 0 or \
            #     micro_bs // (num_gpus_list[num_stages - 1 - i] * 2 // args.max_tp) in args.micro_batch_size):
            node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
            max_tp_group = get_max_tp(args,None,node_id_num[i])
            if num_gpus_list[num_stages - 1 - i] <= num_gpus_remained and \
                (num_gpus_list[num_stages - 1 - i] * 2 // max_tp_group == 0 or \
                micro_bs // (num_gpus_list[num_stages - 1 - i] * 2 // max_tp_group) in args.micro_batch_size):
                num_gpus_remained -=  num_gpus_list[num_stages - 1 - i]
                num_gpus_list[num_stages - 1 - i] *= 2
                found = True
                print(f"update: {num_gpus_list} (inc gpus in stage {num_stages - 1 - i})")
                break
            else:
                print(f"fail to update on stage {num_stages - 1 - i}")
        if not found:
            micro_bs_index += 1
            assert micro_bs_index < len(args.micro_batch_size)
            micro_bs =  args.micro_batch_size[micro_bs_index]
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
        # if num_gpus_list[i] <= args.max_tp:
        if num_gpus_list[i] <= get_max_tp(args,None,node_id_num[i]):
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        else:
            # tp_size_list += [args.max_tp for _ in range(num_ops_per_stage[i])]
            tp_size_list += [get_max_tp(args,None,node_id_num[i]) for _ in range(num_ops_per_stage[i])]
            # dp_size_list += [num_gpus_list[i] // args.max_tp for _ in range(num_ops_per_stage[i])]   
            dp_size_list += [num_gpus_list[i] // get_max_tp(args,None,node_id_num[i]) for _ in range(num_ops_per_stage[i])]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

def generate_imbalance_op_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
            node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
            max_tp_group = get_max_tp(args,None,node_id_num[i])
            if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < max_tp_group and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        return None
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    if args.model_name == "resnet":
        num_ops = len(full_op_list)
        num_layers_list = [1 for _ in range(num_stages)]
        num_layers_list[0] += 33 - sum(num_layers_list)
        num_ops_per_stage = [num_layers_list[i] * 8 for i in range(num_stages)]
        num_ops_per_stage[0] += 4
        num_ops_per_stage[-1] += 2
    elif args.model_name in ["gpt", "scale-layer"]:
        num_ops = len(full_op_list)
        num_layers_list = [1 for _ in range(num_stages)]
        num_layers_list[-1] += args.num_layers - sum(num_layers_list)

        num_ops_per_stage = [num_layers_list[i] * args.num_ops_each_layer for i in range(num_stages)]
        num_ops_per_stage[0] += 1
        num_ops_per_stage[-1] += 2        

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
        dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
    if flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

#deprecated
def generate_initial_config(num_stages, node_order_idx,args):   
    full_op_list = get_full_op_list(args)
    if args.initial_point == "balance":
        return generate_balance_config(full_op_list, num_stages, args)
    elif args.initial_point == "imbalance_gpu":
        return generate_imbalance_gpu_config(full_op_list, num_stages, args)
    elif args.initial_point == "imbalance_op":
        return generate_imbalance_op_config(full_op_list, num_stages, args)    
    elif args.initial_point == "test":
        return generate_test_config(full_op_list, num_stages, args)    
    elif args.initial_point == "balance_v1":
        return generate_balance_config_v1(full_op_list, num_stages, args)
    else:
        with open(args.initial_point, "r") as f:
            config_dict = json.load(f)
        if(config_dict["num_stages"] != num_stages):
            args.initial_point = "balance_v1"
            return generate_initial_config(num_stages,node_order_idx, args)
        elif(args.config_node_order_idx == node_order_idx):
            print(f"read initial point from {args.initial_point}")
            # print(f"read_config_from_json {read_config_from_json(args)}")
            return read_config_from_json(args)     
        else:
            args.initial_point = "balance_v1"
            return generate_initial_config(num_stages,node_order_idx, args)
        

def generate_initial_config_v1(num_stages, node_order_idx,device_group,args): #TODO metis initial config
    full_op_list = get_full_op_list(args)
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0] 
    flex_recompute = args.flex_recompute
    num_gpus_list = device_group
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    num_ops = len(full_op_list)
    # num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    # num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    # print("args.model_name ",args.model_name )
    # print("args.op_grain ",args.op_grain )
    if(args.model_name == "gpt" and args.op_grain == "layer"):
        num_ops_per_stage = [0] * num_stages
        for i in range(len(num_ops_per_stage)):
            num_ops_per_stage[i] = num_ops // num_stages // args.num_ops_each_layer * args.num_ops_each_layer
        num_ops_per_stage[0]+=1
        # num_ops_per_stage[-1] +=num_ops - sum(num_ops_per_stage)
        # print("num_ops_per_stage",num_ops_per_stage)
        num_ops_per_stage[-1] += 2
        num_ops_temp = num_ops - sum(num_ops_per_stage)
        # print("num_ops_temp",num_ops_temp)
        # num_ops_per_stage[-1] += num_ops_temp 

        while(num_ops_temp>0):
            for i in range(len(num_ops_per_stage)-1,-1,-1):
                # print("num_ops_temp",num_ops_temp)
                if(num_ops_temp>0):
                    num_ops_per_stage[i]+= args.num_ops_each_layer
                    num_ops_temp -= args.num_ops_each_layer
                else:
                    break
                       
        print("num_ops_per_stage",num_ops_per_stage)
        print("num_ops",num_ops)
        # print("sum(num_ops_per_stage)",sum(num_ops_per_stage))
    else:
        num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
        num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    tp_size_list = []
    dp_size_list = []
    cp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
            cp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]     
            cp_size_list += [1 for _ in range(num_ops_per_stage[i])]     
    micro_bs =  dp_size_list [0]
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    # print("num_ops_per_stage",num_ops_per_stage)
    # print("dp_size_list",dp_size_list)
    # print("tp_size_list",tp_size_list)
    # print("num_ops",num_ops)
    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, cp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)
    return initial_config


def sort_configs(config_list, sort_metric):
    '''
    按照sort_metric对config_list进行从小到大排序
    使用的sort算法是插入排序
    '''
    new_list = []

    if sort_metric == "max_stage_time":
        for i in range(len(config_list)):
            max_stage_time = max(config_list[i].time_list)
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if max_stage_time < max(new_list[j].time_list):
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break 
                if not inserted:
                    new_list.append(config_list[i])   
    elif sort_metric == "total_gpu_time":
        for i in range(len(config_list)):
            gpu_time = config_list[i].total_gpu_time
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if gpu_time < new_list[j].total_gpu_time:
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break 
                if not inserted:
                    new_list.append(config_list[i])   
    else:
        raise RuntimeError(f"sort_metric {sort_metric} not supported.")
    return new_list

def get_boundary_list(args):
    num_gpus_at_boundary_list = []
    for i in range(0,args.num_nodes):
        if(i):
            num_gpus_at_boundary_list.append(args.num_gpus_per_node_list[i] + num_gpus_at_boundary_list[i-1]) 
        else:
            num_gpus_at_boundary_list.append(args.num_gpus_per_node_list[0])
    return num_gpus_at_boundary_list





def check_legality(config, args):
    num_gpus_from_start = []
    num_gpus = 0
    for i in range(config.num_stages):
        if config.stages[i].num_gpus not in [1, 2, 4, 8]:
            return False
        num_gpus += config.stages[i].num_gpus
        num_gpus_from_start.append(num_gpus)
    if num_gpus != args.num_gpus:
        return False
    # num_gpus_at_boundary_list = [args.num_gpus_per_node * i for i in range(1, args.num_nodes + 1)]
    
    num_gpus_at_boundary_list = get_boundary_list(args)

    for num_gpus_at_boundary in num_gpus_at_boundary_list:
        if num_gpus_at_boundary not in num_gpus_from_start:
            return False 
    return True

def format_size_list(size_list):
    output_string = "["
    for i in range(len(size_list)):
        max_val = max(size_list[i])
        min_val = min(size_list[i])
        if min_val == max_val:
            output_string += f"{max_val}, "
        else:
            output_string += f"{min_val}~{max_val}, "
    output_string += "]"
    return output_string

def print_simple_config_info(config, info="", add_history=False, print_recompute_ops=False, print_debug_info=False):
    if config is None:
        return

    num_ops_stage, tp_per_stage, dp_per_stage, recompute_ops, base_batch_size, algo_per_stage = config_details(config)
    gpu_list = [config.stages[i].num_gpus for i in range(config.num_stages)]
    recompute_ops_sum = []
    for i in range(config.num_stages):
        recompute_ops_sum.append(sum(recompute_ops[i]))
    detailed_tp_size = format_size_list(tp_per_stage)
    detailed_dp_size = format_size_list(dp_per_stage)
    detailed_algos = format_size_list(algo_per_stage)

    history = "{}|{:.2f}|{:.2f}| op = {}| tp = {} | dp = {} | algo = {} | rc = {} | gpus = {} | micro_bs = {} | time = {} | memory = {}".format(
        info, max(config.time_list)/1000, max(config.memory_list), num_ops_stage, detailed_tp_size, detailed_dp_size, detailed_algos, recompute_ops_sum, gpu_list, base_batch_size, list(map(int, config.time_list)), list(map(int, config.memory_list)))
    debug_info(history, print_debug_info)
    if add_history:
        config.history += history + "\n"

    if print_recompute_ops:
        for i in range(config.num_stages):
            ops = config.stages[i].ops
            recompute_ops = config.stages[i].recompute_ops
            stage_recompute_index_string = f"[stage {i} recompute_ops] "
            stage_string = f"stage {i}: "
            for j in range(len(recompute_ops)):
                if recompute_ops[j] == 1:
                    stage_recompute_index_string += f"{j},"
                    stage_string += ops[j] + ", "
            print(stage_recompute_index_string)
            print(stage_string)

        for i in range(config.num_stages):
            ops = config.stages[i].ops
            algo_list = config.stages[i].algo
            stage_algo_string = f"[stage {i} algo1] "
            for j in range(len(algo_list)):
                if algo_list[j] == 1:
                    stage_algo_string += f"[{j}] {ops[j]} "
            print(stage_algo_string)

    return 

def is_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        return False
    else:
        if target in visited_set[hash_str]:
            return True 
        else:
            return False

def mark_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        visited_set[hash_str] = [target]
    else:
        visited_set[hash_str].append(target)

def num_visited(visited_set, hash_str):
    if hash_str not in visited_set:
        return  0
    else:
        return len(visited_set[hash_str])

def save_search_trend_in_csv(search_time_list, exec_time_list, file_name):
    result_list = [["search_time (s)", "config time (ms)"]]
    assert len(search_time_list) == len(exec_time_list)
    for i in range(len(search_time_list)):
        new_line = [search_time_list[i], exec_time_list[i]]
        result_list.append(new_line)
    
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result_list)


# search 输出top_count个结果
def print_config_info(result_dict_result,args,stage_num, device_group_idx, node_order_idx,top_count,config_dict):
        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown ,each_stage_memory_breakdown,min_get_best_config_time ,best_config= result_dict_result
        config_thpt = args.global_batch_size / (config_time/1000)
        data = []
        data_1_1 = []
        data_1_2 = []
        data_1 = []
        data_2 = []
        data_3 = []

        data.append([stage_num, f"{config_time/1000:.2f}", f"{config_thpt:.2f}", f"{config_mem:.0f}", f"{explored_cases}"])
        data_1_1.append([f"{args.node_order_list[node_order_idx]}" , f"{node_order_idx}" ,f"{device_group_idx}"])
        data_1_2.append([f"{args.total_device_group_list[stage_num][device_group_idx]}"])

        for time_idx  in range(len(time_list)) :
            time_list[time_idx] = time_list[time_idx]/1000


        time_list = format_list(time_list)  

        # for memory_idx  in range(len(memory_list)) :
        #     memory_list[memory_idx] = memory_list[memory_idx]/1000

        memory_list = format_list(memory_list)  
        eff_loss_time_per_gpu = format_list(eff_loss_time_per_gpu)  
        recomp_time_per_gpu = format_list(recomp_time_per_gpu)  
        data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])


        for stage_list in each_stage_time_breakdown:
            for time_idx in range(1,len(stage_list)):
                stage_list[time_idx] = stage_list[time_idx]/1000


        each_stage_time_breakdown = [format_list(stage_list)  for stage_list in each_stage_time_breakdown]
        data_2.append([f"{each_stage_time_breakdown}"])
        each_stage_memory_breakdown = [format_list(stage_list)  for stage_list in each_stage_memory_breakdown]
        data_3.append([f"{each_stage_memory_breakdown}"])
        print(f" ============================== TOP {top_count} ==============================")
        header = ["# of stages", "est_iteration_time(s)", "est_thpt(samples/s)", "est_mem(MB)", "# of explored cases" ] 
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["Node Order" , "node_order_idx" ,"device_group_idx"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1_1:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["Device Group"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1_2:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["adaptive_times", "time_list (s)", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header= ["each_stage_time_breakdown [ total (ms) fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_2:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
                
        header= ["each_stage_memory_breakdown [total (MB) memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_3:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["# min_get_best_config_time"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        print(min_get_best_config_time)




        stage_idx =0
        num_stage = stage_num
        comm_table=[[ 0 for __ in range(num_stage) ] for _ in range(num_stage)]
        r=[[ True for __ in range(num_stage) ] for _ in range(num_stage)]
        # print("config.each_stage_time_breakdown")
        fwd_per_stage=[]
        bwd_per_stage=[]

        for time_ in each_stage_time_breakdown:
            fwd_per_stage.append(time_[1]/1000)
            bwd_per_stage.append((time_[2]+time_[3])/1000) 
            if(stage_idx<num_stage-1):
                comm_table[stage_idx][stage_idx+1] = time_[5]/1000
                comm_table[stage_idx+1][stage_idx] = time_[5]/1000
            stage_idx+=1

        config_ = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=comm_table, r=r, p=num_stage, m=config_dict["global_batch_size"]//config_dict["micro_batch_size"], v=1, c=0, overlap_c=True)
        og = OneFOneBGenerator(config_)
        operations = og.generate()
        executor = OperationExecutor(config_, operations)
        executor.execute()
        ans = executor.makespan()
        print("simulator time " , ans)


def format_list(list):
    formatted_list = [float(f"{x:.3f}") for x in list]
    return formatted_list

def save_and_print_top_configs(result_dict, args):
    # print("result_dict",result_dict)
    sorted_time_list = [MAX_VALUE]
    sorted_config_stage_list = [(MAX_VALUE, MAX_VALUE, MAX_VALUE)]

    sorted_config_stage_list_each_stage = (MAX_VALUE, MAX_VALUE, MAX_VALUE)
    for i in range(args.start_num_stages, args.end_num_stages + 1):
        best_stage_time = MAX_VALUE
        for node_order_idx in range(len(args.node_order_list)):
            for device_group_idx in range(len(args.total_device_group_list[i])):
        # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                    config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                    if config_time > 0 and best_stage_time > config_time:
                        best_stage_time = config_time
                        sorted_config_stage_list_each_stage = (i, device_group_idx, node_order_idx)

        if(best_stage_time!=MAX_VALUE):
            print(f"stage_num {i}")
            stage_num, device_group_idx, node_order_idx = sorted_config_stage_list_each_stage
            src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{i}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
            with open(src_file, "r") as f:
                config_dict = json.load(f)
            print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,-1,config_dict)
        else:
            print(f"stage_num {i} is None")

        
                            
    for node_order_idx in range(len(args.node_order_list)):
        for i in range(args.start_num_stages, args.end_num_stages + 1):
                for device_group_idx in range(len(args.total_device_group_list[i])):
            # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                    if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                        if config_time > 0:
                            for j in range(len(sorted_time_list)):
                                current_config_time = sorted_time_list[j]
                                if config_time < current_config_time:
                                    sorted_time_list.insert(j, config_time)
                                    sorted_config_stage_list.insert(j, (i, device_group_idx, node_order_idx))
                                    # print(f"insert {i}stages, device_group_idx = {device_group_idx}, node_order_idx = {node_order_idx} to {j}th")
                                    break
    sorted_config_stage_list.pop()
    sorted_time_list.pop()
    # print("sorted_config_stage_list",sorted_config_stage_list)
    ##### save configs:


    save_count = 0
    top_count = 0
    for i in range(min((args.end_num_stages - args.start_num_stages + 1)*len(args.node_order_list), args.num_of_show_configs, len(sorted_time_list))): #TODO 总数需修改加上dg

        stage_num, device_group_idx, node_order_idx = sorted_config_stage_list[i]
        # print(f"stage_num = {stage_num}, node_order_idx = {node_order_idx}")
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'
        
        if(save_count < args.num_of_saved_configs):
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                save_count += 1
                print(f"best config save to {dst_file} ")
                print(f"best config save to {src_file} ")
            else:
                print(f"file {src_file} does not exist.")
        else:
            print(f"TOP {top_count} config save to {src_file}")
        with open(src_file, "r") as f:
            config_dict = json.load(f)
        print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,top_count,config_dict)
        top_count += 1


def save_and_print_top_configs_dp_hete(result_dict, args ,start_num_stages ,end_num_stages):
    # print("result_dict",result_dict)
    sorted_time_list = [MAX_VALUE]
    sorted_config_list = [MAX_VALUE]
    sorted_config_stage_list = [(MAX_VALUE, MAX_VALUE, MAX_VALUE)]
    sorted_config_stage_list_each_stage = (MAX_VALUE, MAX_VALUE, MAX_VALUE)
    print("total_device_group_list",args.total_device_group_list)
    print("start_num_stages",start_num_stages)
    print("end_num_stages",end_num_stages)
    total_explored_cases = 0
    for i in range(start_num_stages, end_num_stages + 1):
        best_stage_time = MAX_VALUE
        for node_order_idx in range(len(args.node_order_list)):
            for device_group_idx in range(len(args.total_device_group_list[i])):
                if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                    config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                    total_explored_cases+=explored_cases
                    if config_time > 0 and best_stage_time > config_time:
                        best_stage_time = config_time
                        sorted_config_stage_list_each_stage = (i, device_group_idx, node_order_idx)

        if(best_stage_time!=MAX_VALUE):
            print(f"stage_num {i}")
            stage_num, device_group_idx, node_order_idx = sorted_config_stage_list_each_stage
            src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{i}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
            with open(src_file, "r") as f:
                config_dict = json.load(f)
            print_config_info(result_dict[f"{node_order_idx}_{i}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,-1,config_dict)
        else:
            print(f"stage_num {i} is None")
    print("total_explored_cases",total_explored_cases)
    for node_order_idx in range(len(args.node_order_list)):
        for i in range(start_num_stages, end_num_stages + 1):
                for device_group_idx in range(len(args.total_device_group_list[i])):
            # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                    if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                        # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                        if config_time > 0:
                            for j in range(len(sorted_time_list)):
                                current_config_time = sorted_time_list[j]
                                if config_time < current_config_time:
                                    sorted_time_list.insert(j, config_time)
                                    sorted_config_stage_list.insert(j, (i, device_group_idx, node_order_idx))
                                    sorted_config_list.insert(j, best_config)
                                    # print(f"insert {i}stages, device_group_idx = {device_group_idx}, node_order_idx = {node_order_idx} to {j}th")
                                    break
    sorted_config_stage_list.pop()
    sorted_time_list.pop()
    sorted_config_list.pop()
    # print("sorted_config_stage_list",sorted_config_stage_list)
    ##### save configs:
    save_count = 0
    top_count = 0

    best_config = None
    best_config_json = None

    for i in range(min((end_num_stages - args.start_num_stages + 1)*len(args.node_order_list), args.num_of_show_configs, len(sorted_time_list))): #TODO 总数需修改加上dg

        stage_num, device_group_idx, node_order_idx = sorted_config_stage_list[i]
        # print(f"stage_num = {stage_num}, node_order_idx = {node_order_idx}")
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'
        
        if(save_count < args.num_of_saved_configs):
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                save_count += 1
                print(f"best config save to {dst_file}")
            else:
                print(f"file {src_file} does not exist.")
        else:
            print(f"TOP {top_count} config save to {src_file}")

        with open(src_file, "r") as f:
            config_dict = json.load(f)
            if(best_config_json is None):
                best_config_json  = config_dict
                best_config =  sorted_config_list[i]
        print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,top_count,config_dict)
        top_count += 1


    return  best_config, best_config_json


# cost prediction
def save_and_print_configs(config:AcesoConfig , args):

    ##### save configs:
    data = []
    data_1 = []
    data_2 = []
    data_3 = []

    stage_num = len(config.stages)
    config_time = max(config.time_list)/1000
    config_mem = max(config.memory_list)
    adaptive_times = config.adaptive_times
    time_list = config.time_list.copy()
    memory_list = config.memory_list

    for i in range(len(memory_list)):
        memory_list[i]/=1024

    for i in range(len(time_list)):
        time_list[i]/=1000



    eff_loss_time_per_gpu = config.breakdown_eff_loss_time_per_gpu
    recomp_time_per_gpu = config.breakdown_recomp_time_per_gpu
    each_stage_time_breakdown = config.each_stage_time_breakdown.copy()
    each_stage_memory_breakdown = config.each_stage_memory_breakdown

    for i in range(stage_num):
        for j in range(1,len(each_stage_time_breakdown[i])):
            each_stage_time_breakdown[i][j]/=1000

    if(0): #TODO 暂时不要保存结果
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
    config_thpt = args.global_batch_size / (config_time)
    data.append([stage_num,f"{config_time}",f"{config_thpt:.2f}", f"{config_mem:.0f}" ,f"{args.global_batch_size}",f"{config.micro_bs}"] )
    data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])
    data_2.append([f"{each_stage_time_breakdown}"])
    data_3.append([f"{each_stage_memory_breakdown}"])

    header = ["# of stages", "est_iteration_time(s)", "est_thpt(samples/s)", "est_mem(MB)" ,"global_bs" ,"micro_bs"] 
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")
    header = ["adaptive_times", "time_list", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_1:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    header= ["each_stage_time_breakdown [total (/1000), fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for list in each_stage_time_breakdown:
        print(list)
    print("\n")

    header= ["each_stage_memory_breakdown [total , memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for list in each_stage_memory_breakdown:
        print(list)
    print("\n")
    # for row in data_3:
    #     print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

# pipeline output
def print_configs(config:AcesoConfig , args):

    ##### save configs:
    data = []
    data_1 = []
    data_2 = []
    data_3 = []

    stage_num = len(config.stages)
    config_time = max(config.time_list)
    config_mem = max(config.memory_list)
    config_gbs = config.global_bs
    config_mbs = config.micro_bs
    node_info = config.node_info
    adaptive_times = config.adaptive_times
    time_list = config.time_list
    # memory_list = config.memory_list
    memory_list = config.memory_list.copy()
    each_stage_time_breakdown = config.each_stage_time_breakdown.copy()
    gpu_num = 0
    Tflops = 0
    for key in node_info.keys():
        gpu_num += node_info[key]['GPU_NUM']
        Tflops +=  args.device_info[node_info[key]['GPU']]['TFLOPS'] * gpu_num


    for i in range(len(memory_list)):
        memory_list[i]/=1024

    for time_breakdown in each_stage_time_breakdown:
        for i in range(1,len(time_breakdown)):
            time_breakdown[i]/=1000

    eff_loss_time_per_gpu = config.breakdown_eff_loss_time_per_gpu
    recomp_time_per_gpu = config.breakdown_recomp_time_per_gpu
    # each_stage_time_breakdown = config.each_stage_time_breakdown
    each_stage_memory_breakdown = config.each_stage_memory_breakdown





    print(f"node_info {node_info}")
    config_thpt = config_gbs / (config_time)
    data.append([stage_num,f"{config_time}",f"{config_thpt:.2f}", f"{config_mem:.0f}", f"{config_gbs}", f"{config_mbs}" , f"{gpu_num}" , f"{Tflops:.2f}" , f"{config_thpt/Tflops * 1000000:.2f}"])
    data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])
    data_2.append([f"{each_stage_time_breakdown}"])
    data_3.append([f"{each_stage_memory_breakdown}"])

    # with open('/home/ymj/project/Aceso/log/motivation_exp/B2x2_Ax2/config_time_hete_gpt_search_n2_g2_2_6B_test_dp_numkater16.log', 'a') as f:
    #     # print("config_time",config_time)
    #     f.write(f"{config_time}\n")  #文件的写操作

    header = ["# of stages", "est_iteration_time(s)", "est_thpt(samples/s)", "est_mem(MB)", "global_bs", "micro_bs" , "GPU_NUM" , "Total Tflops" , "Thpt/TFLOPS *1000"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")
    header = ["adaptive_times", "time_list", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_1:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    # with open('/home/ymj/project/Aceso/log/motivation_exp/B2x2_Ax2/each_stage_time_breakdown_hete_gpt_search_n2_g2_2_6B_test_dp_numlayer16.log', 'a') as f:
    #     # print("config_time",config_time)
    #     for i in range(len(each_stage_time_breakdown)):
    #         print("each_stage_time_breakdown",each_stage_time_breakdown)
    #         f.write(f"{each_stage_time_breakdown[i][0]}\n")  #文件的写操作

    header= ["each_stage_time_breakdown [total (/1000), fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_2:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    header= ["each_stage_memory_breakdown [total , memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))

    for row in data_3:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

    stage_idx =0
    num_stage = stage_num
    comm_table=[[ 0 for __ in range(num_stage) ] for _ in range(num_stage)]
    r=[[ True for __ in range(num_stage) ] for _ in range(num_stage)]
    # print("config.each_stage_time_breakdown")
    fwd_per_stage=[]
    bwd_per_stage=[]

    for time_ in each_stage_time_breakdown:
        fwd_per_stage.append(time_[1]/1000)
        bwd_per_stage.append((time_[2]+time_[3])/1000) 
        if(stage_idx<num_stage-1):
            comm_table[stage_idx][stage_idx+1] = time_[5]/1000
            comm_table[stage_idx+1][stage_idx] = time_[5]/1000
        stage_idx+=1

    config_ = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=comm_table, r=r, p=num_stage, m=config_gbs//config_mbs, v=1, c=0, overlap_c=True)
    og = OneFOneBGenerator(config_)
    operations = og.generate()
    executor = OperationExecutor(config_, operations)
    executor.execute()
    ans = executor.makespan()
    print("OneFOneBGenerator simulator time " , ans)

def save_and_print_top_hybrid_config (hybrid_config_list: List[HybridConfig],args):
    
    #根据hybrid_config_list 中 每一个HybridConfig 的 max(每一个HybridConfig.time_list_pipline)按从小到大进行排序
    pairs = [(max(config.time_list_pipline), config) for config in hybrid_config_list]
    # 按max_time从小到大排序
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    # 分离出排序后的配置和对应的时间列表
    sorted_configs = [config for max_time, config in sorted_pairs]
    sorted_max_times = [max_time for max_time, config in sorted_pairs]

    for i in range(min(5, len(sorted_configs))):
        dump_hybrid_config_to_json(sorted_configs[i], f'{args.config_save_path}{args.model_name}_{args.model_size}_NUMPIPE{sorted_configs[i].num_pipelines}_TOP{i}_{args.config_suffix}.json', args)
        print(f"=================TOP DP HETE {i}===================")
        for j in range(sorted_configs[i].num_pipelines):
            print(f"=================TOP DP HETE PIPELINE {j}===================")
            print_configs(sorted_configs[i].pipelines[j], args)






    return sorted_configs , sorted_max_times


def print_search_details(config, args, num_stages, node_order_idx,device_group_idx,num_targets_list, num_hops_list, search_time_list, config_time_list, reserved_mem_list, num_explored_cases):
    if(args.reduce_output or args.test_search_time):
        return
    print(f"\n========== Best Result (num_stages = {num_stages}) node_order_idx ={node_order_idx} device_group_idx = {device_group_idx} ==========")
    if config is not None:
        print_simple_config_info(config, print_recompute_ops=False, print_debug_info=False)
        print(config.history)
        print(f"num_targets: {num_targets_list}")
        print(f"num_hops: {num_hops_list}") 
        save_config_info_to_csv(config, reserved_mem_list, f'{args.config_save_path}csv/info_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv')  
        save_distribution_to_csv(num_targets_list, num_hops_list, f'{args.log_path}trends/distribution_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv')       
    else:
        print("No feasible solution.")

    sum_time = 0
    accum_list = []
    for i in range(len(search_time_list)):
        if i == 0:
            accum_list.append(0)
        else:
            time_one_trial = search_time_list[i]
            sum_time += int(time_one_trial+1)
            accum_list.append(sum_time)

    print(f"search time = {sum(search_time_list):.2f} s.  {accum_list} \nnum_explored_cases = {num_explored_cases}")
    # save_search_trend_in_csv(accum_list, config_time_list, f"{args.log_path}trends/{args.model_name}_{args.model_size}_{num_stages}stages_init_{args.initial_point}_{args.max_num_hops}hops_{args.config_suffix}.csv")
    print(f"time trend: {list(map(int, config_time_list))}\n")

## some globally used values
timers = Timers()



def get_memory_limit(args,node_id_num):
    min_memory = MAX_VALUE
    for node_id, num in node_id_num.items(): #TODO can be optimized
        min_memory = min(min_memory, args.device_info[node_id]["memory"])
    # print(f"min_memory: {min_memory}")
    return min_memory

def check_memory_legacy(args,config):
    assert len(config.memory_list) == len(config.stages) , f"memory list length not match, which {len(config.memory_list)} != {len(config.stages)}"

    for i in range(config.num_stages):
        min_memory = get_memory_limit(args,config.stages[i].node_id_num)
        # print(f"stage {i}, config.memory_list[i] {config.memory_list[i]}, min_memory: {min_memory}")
        if min_memory < config.memory_list[i]:
            return False
    return True

def get_max_tp(args,stage:AcesoStageInfo,node_id_num =None): #TODO when tensor parallelism can be  split different workloads to different nodes
    max_tp_group = MAX_VALUE

    if(node_id_num is not None):
        for node_id, num in node_id_num.items():
            max_tp_group = min(max_tp_group,num)
        
        return max_tp_group

    for node_id, num in stage.node_id_num.items():
        max_tp_group = min(max_tp_group,num)
    return min(max_tp_group,args.max_tp)


def gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list): # gpu_num_list = [1,2,4,8] node_id_num = {0:{A100:1},1:{A100:2},2:{A100:4},3:{A100:8}}
    num_gpus_per_node_list = args.num_gpus_per_node_list
    gpu_type_list = args.gpu_type_list 
    node_id_num = {}
    current_node_id = 0
    current_gpu_id = 0
    # print(gpu_type_list,num_gpus_list,num_gpus_per_node_list)
    for i in range(len(num_gpus_list)):
        num_gpus = num_gpus_list[i]
        node_id_num[i] = {}
        while num_gpus > 0:
            if(gpu_type_list[current_node_id] not in node_id_num[i].keys()):
                node_id_num[i][gpu_type_list[current_node_id]] = 0
            add = min(num_gpus, num_gpus_per_node_list[current_node_id]-current_gpu_id)
            node_id_num[i][gpu_type_list[current_node_id]] += add
            num_gpus -= add
            current_gpu_id += add
            if current_gpu_id == num_gpus_per_node_list[current_node_id]:
                current_gpu_id = 0
                current_node_id += 1
    return node_id_num



    return each_stage_gpu_type_num

def generate_permutations(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    permutations = list(itertools.permutations(values))
    
    unique_permutations = []
    seen = set()
    
    for perm in permutations:
        perm_tuple = tuple(sorted((k, frozenset(v.items())) for k, v in zip(keys, perm)))
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            perm_dict = {keys[i]: perm[i] for i in range(len(keys))}
            unique_permutations.append(perm_dict)
    
    return unique_permutations



class BipartiteMatcher:
    def __init__(self, a_list, b_list):
        self.a_list = a_list
        self.b_list = b_list
        self.adj = [[] for _ in range(len(a_list))]
        for a_idx, a_gpu in enumerate(a_list):
            for b_idx, b_gpu in enumerate(b_list):
                if b_gpu[0] >= a_gpu[0] and b_gpu[1] >= a_gpu[1]:
                    self.adj[a_idx].append(b_idx)
        self.match_to = [-1] * len(b_list)
    
    def find_max_matching(self):
        result = 0
        for a in range(len(self.a_list)):
            visited = [False] * len(self.b_list)
            if self.dfs(a, visited):
                result += 1
        return result
    
    def dfs(self, a, visited):
        for b in self.adj[a]:
            if not visited[b]:
                visited[b] = True
                if self.match_to[b] == -1 or self.dfs(self.match_to[b], visited):
                    self.match_to[b] = a
                    return True
        return False

def solution(A, B):
    a_gpus = [(gpu['tflops'], gpu['memory']) for gpu in A.values()]
    b_gpus = [(gpu['tflops'], gpu['memory']) for gpu in B.values()]
    
    if len(b_gpus) < len(a_gpus):
        return False
    
    matcher = BipartiteMatcher(a_gpus, b_gpus)
    max_match = matcher.find_max_matching()
    
    return max_match == len(a_gpus)


def get_upper_bound(pipeline_split,cache_thp,prex2dict,pipeline_split_start,args):

    determinated_upper_bound = [0 ] * pipeline_split_start
    approximate_upper_bound = [0 ] * pipeline_split_start

    def get_tflops_rate(A_gpu_set,B_gpu_set):
        memory_A = 0
        memory_B = 0 
        tflops_A =0 
        tflops_B =0 
        for key in A_gpu_set.keys():
            memory_A+=A_gpu_set[key]["memory"]
            tflops_A+=A_gpu_set[key]["tflops"]
        for key in B_gpu_set.keys():
            memory_B+=B_gpu_set[key]["memory"]
            tflops_B+=B_gpu_set[key]["tflops"]

        return max(memory_A*tflops_A/memory_B/tflops_B,1)
    
    A_gpu_set = {}
    for pipe_idx in range(pipeline_split_start,len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline
        determinated_upper_bound.append(10000)
        approximate_upper_bound.append(10000)
        gpu_type_num_dict_ = pipeline_split[pipe_idx]
        gpu_type_num_dict = {}
        idx = 0
        for item in gpu_type_num_dict_:
            gpu_type_num_dict[f"{idx}"] = item
            idx += 1
        for idx in gpu_type_num_dict.keys():
            A_gpu_set[idx] = gpu_type_num_dict[idx]
            A_gpu_set[idx]["memory"] = args.device_info[A_gpu_set[idx]["GPU"]]["memory"]
            A_gpu_set[idx]["tflops"] = args.device_info[A_gpu_set[idx]["GPU"]]["TFLOPS"]

        for key in cache_thp.keys():
            B_gpu_set = {}

            for idx in prex2dict[key].keys():
                B_gpu_set[idx] = prex2dict[key][idx]
                B_gpu_set[idx]["memory"] = args.device_info[B_gpu_set[idx]["GPU"]]["memory"]
                B_gpu_set[idx]["tflops"] = args.device_info[B_gpu_set[idx]["GPU"]]["TFLOPS"]

            approximate_upper_bound[pipe_idx] = min(args.upper_bound_rate *get_tflops_rate(A_gpu_set,B_gpu_set)* cache_thp[key], approximate_upper_bound[pipe_idx] ) 
            if(solution(A_gpu_set,B_gpu_set)):
                # print("A_gpu_set",A_gpu_set ,"gpu_type_num_dict",gpu_type_num_dict, "B_gpu_set",B_gpu_set)
                determinated_upper_bound[pipe_idx] = min(cache_thp[key], determinated_upper_bound[pipe_idx] ) 
    
    result = [ min(determinated_upper_bound[pipe_idx] , approximate_upper_bound[pipe_idx]) for pipe_idx in range(len(pipeline_split))]
    return result, determinated_upper_bound,approximate_upper_bound


# A =  {'0': { 'tflops': 8 , 'memory': 10 }, '1': { 'tflops': 18 , 'memory': 20}, '2': { 'tflops': 18 , 'memory': 20},'3': { 'tflops': 8 , 'memory': 10 } }              
# B = {'0': { 'tflops': 18 , 'memory': 20 }, '1': {'tflops': 9 , 'memory': 10}, '2': { 'tflops': 1 , 'memory': 1}    }                  
import numpy as np
def sort_pipeline_split_list(pipeline_split_list,args):
    if(not args.sort_pipeline):
        return pipeline_split_list
    loss_list = []
    var_inter_gpu_list = []
    var_intra_gpu_list = []
    for pipeline_split in pipeline_split_list:
        inter_gpu = []
        var_intra_gpu = 0
        for pipe_idx in range(len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline

            intra_gpu = []
            gpu_type_num_dict_ = pipeline_split[pipe_idx]
            gpu_type_num_dict = {}
            idx = 0
            for item in gpu_type_num_dict_:
                gpu_type_num_dict[f"{idx}"] = item
                idx += 1
            for idx in gpu_type_num_dict.keys():
                for num_gpu in range(gpu_type_num_dict[idx]["GPU_NUM"]):
                    intra_gpu.append(args.device_info[gpu_type_num_dict[idx]["GPU"]]["TFLOPS"] )
            # print("intra_gpu",intra_gpu)
            inter_gpu.append(sum(intra_gpu))
            var_intra_gpu += np.var(intra_gpu)
        var_inter_gpu = np.var(inter_gpu)
        loss = var_intra_gpu/args.num_layers + var_inter_gpu * len(pipeline_split) / (args.global_batch_size)
        loss_list.append(loss)
        var_inter_gpu_list.append(var_inter_gpu * len(pipeline_split) / (args.global_batch_size))
        var_intra_gpu_list.append(var_intra_gpu/args.num_layers)
        # print("pipeline_split",pipeline_split)
        # print("loss",loss)
        print("var_inter_gpu",var_intra_gpu)
        print("var_intra_gpu", var_inter_gpu)
    sorted_pipeline_split_list = [x for _, x in sorted(zip(loss_list, pipeline_split_list))]
    sorted_var_intra_gpu_list = [x for _, x in sorted(zip(loss_list, var_intra_gpu_list))]
    sorted_var_inter_gpu_list = [x for _, x in sorted(zip(loss_list, var_inter_gpu_list))]
    sorted_loss_list = [x for x, _ in sorted(zip(loss_list, pipeline_split_list))]

    # for i in range(len(sorted_pipeline_split_list)):
    #     print(f"loss {sorted_loss_list[i]}")
    #     print(f" {sorted_pipeline_split_list[i]}")
    #     print(f"sorted_loss_list {sorted_loss_list[i]}")
    #     print(f"sorted_var_intra_gpu_list {sorted_var_intra_gpu_list[i]}")
    #     print(f"sorted_var_inter_gpu_list {sorted_var_inter_gpu_list[i]}")

    # exit()

    return sorted_pipeline_split_list


def generate_permutations_large(input_dict):

    input_dict

    keys = list(input_dict.keys())
    values = list(input_dict.values())
    permutations = list(itertools.permutations(values))
    
    unique_permutations = []
    seen = set()
    
    for perm in permutations:
        perm_tuple = tuple(sorted((k, frozenset(v.items())) for k, v in zip(keys, perm)))
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            perm_dict = {keys[i]: perm[i] for i in range(len(keys))}
            unique_permutations.append(perm_dict)
            print("perm_dict",perm_dict)

    return unique_permutations


def test():
    values = [1]*8 +[0]*2
    permutations = list(itertools.permutations(values))
    print(len(permutations))

def unique_permutations(input_dict):

    input_list = []
    for key in input_dict.keys():
        input_list+= [key]*input_dict[key]["NODE_NUM"]



    s = sorted(input_list)



    n = len(s)
    res = []
    used = [False] * n  # 标记字符是否被使用过

    def backtrack(path):
        if len(path) == n:
            res.append(path)
            return
        for i in range(n):
            # 跳过已使用的字符
            if used[i]:
                continue
            # 剪枝条件：当前字符与前一个相同且前一个未被使用时跳过
            if i > 0 and s[i] == s[i-1] and not used[i-1]:
                continue
            used[i] = True
            backtrack(path + [s[i]])
            used[i] = False

    backtrack([])

    res_ = []
    for item in res :
        item_ = []
        for i in item:
            item_.append({"GPU": input_dict[i]["GPU"], "GPU_NUM": input_dict[i]["GPU_NUM"]})
        res_.append(item_)

    return res_

# 示例用法

def test():
    num_gpus =256
    for num_stages in range(2,32):
        device_group_list = find_combinations_v1(num_stages,num_gpus, num_gpus//num_stages, num_gpus//num_stages)



def split_devices_to_pipeline(input_list, n):

    input_list_ = input_list
    input_list = []
    for key in input_list_.keys():
        input_list.append(input_list_[key])


    def partition(elements, n):
        if n == 1:
            yield [elements]
            return
        if len(elements) == n:
            yield [[e] for e in elements]
            return
        first = elements[0]
        rest = elements[1:]
        # Case 1: first is in its own subset
        for p in partition(rest, n-1):
            yield [[first]] + p
        # Case 2: add first to a subset in each partition of rest into n subsets
        for p in partition(rest, n):
            for i in range(len(p)):
                new_p = [lst.copy() for lst in p]
                new_p[i].append(first)
                yield new_p

    if n <= 0 or n > len(input_list):
        return []

    # Generate all possible partitions
    try:
        all_partitions = list(partition(input_list, n))
    except:
        return []

    # Normalize each partition and deduplicate
    seen = set()
    unique_partitions = []
    for p in all_partitions:
        # Sort each subset by 'GPU' and convert to tuple of tuples
        sorted_subsets = [tuple(sorted((tuple(d.items()) for d in sub), key=lambda x: x[0])) for sub in p]
        # Sort the list of subsets to eliminate order differences between subsets
        sorted_subsets.sort(key=lambda x: (len(x), x))
        # Convert to a tuple for hashing
        key = tuple(sorted_subsets)
        if key not in seen:
            seen.add(key)
            # Convert back to list of lists of dicts
            unique_partitions.append([[dict(pair) for pair in sub] for sub in sorted_subsets])
    return unique_partitions


def idx_to_gpu_type_num_dict(i,j, total_gpu_type_num_dict):

    gpu_type_num_dict = {}
    for k in range(i,j):
        gpu_type_num_dict[f"{k-i}"] = total_gpu_type_num_dict[f"{k}"]

    return gpu_type_num_dict

def get_cache_thpt(gpu_type_num_dict,cache_res):
    return 0


def split_devices_to_pipeline_large(input_dict, n):
    # 将输入字典按键的数值顺序排序，并提取对应的值列表
    sorted_keys = sorted(input_dict.keys(), key=lambda x: int(x))
    values = [input_dict[key] for key in sorted_keys]
    k = len(values)
    
    # 处理无法分割的情况
    if n <= 0 or n > k:
        return []
    
    # 生成所有可能的分割点组合
    split_combinations = itertools.combinations(range(k-1), n-1)
    all_splits = []
    all_splits_point  = []
    for split_points in split_combinations:
        current = 0
        split_result = []
        for point in split_points:
            split_result.append(values[current:point+1])
            current = point + 1
        # 添加最后一个子列表
        split_result.append(values[current:])
        all_splits.append(split_result)
        # 记录分割点
        all_splits_point.append(list(split_points))
    return all_splits 


def max_fun_sum_with_splits(n, k, arr):
    if k <= 0:
        return 0, []
    if k == 1:
        return arr[1][n], []
    cache_sum = []
    cache_split = []
    # 初始化动态规划数组和前驱记录
    dp_prev = [0] * (n + 1)
    for j in range(1, n + 1):
        dp_prev[j] = arr[1][j]
    
    prev_history = []  # 记录每一层的前驱分割点
    count = 0
    for m in range(2, k + 1):
        dp_current = [-float('inf')] * (n + 1)
        prev_current = [0] * (n + 1)  # 当前层的前驱分割点
        
        for j in range(m, n + 1):
            max_val = -float('inf')
            best_i = -1
            # 寻找最优分割点i
            for i in range(1, j):
                count +=1
                current_sum = dp_prev[i] + arr[i][j]
                if current_sum > max_val:
                    max_val = current_sum
                    best_i = i
            dp_current[j] = max_val
            prev_current[j] = best_i
        
        prev_history.append(prev_current)
        dp_prev = dp_current.copy()
        cache_sum.append(dp_prev[n])
        cache_split.append(prev_current)
    max_sum = dp_prev[n]
    
    # 回溯分割点
    split_points = []
    current_j = n
    # 从最后一段开始逆向追踪分割点
    for i in range(k-1):
        m_index = (k-2) - i  # 确定对应的前驱数组层
        if m_index < 0 or m_index >= len(prev_history):
            break
        prev_array = prev_history[m_index]
        split_i = prev_array[current_j]
        if split_i <= 0:
            break
        split_points.append(split_i)
        current_j = split_i
    
    split_points.reverse()  # 反转得到正确的顺序
    return max_sum, split_points


def max_fun_sum(n, k,arr):

    if k == 0:
        return 0
    # 初始化dp_prev数组，dp_prev[j]表示分成1段时前j个元素的最大和
    dp_prev = [0] * (n + 1)
    for j in range(1, n + 1):
        dp_prev[j] = arr[1][j]
    if k == 1:
        return dp_prev[n]
    
    # 动态规划处理每个段数m从2到k
    for m in range(2, k + 1):
        dp_current = [-float('inf')] * (n + 1)
        # j至少为m，因为分成m段至少需要m个元素
        for j in range(m, n + 1):
            max_val = -float('inf')
            # 遍历所有可能的前一段结束点i（i < j）
            for i in range(1, j):
                current_sum = dp_prev[i] + arr[i][j]
                if current_sum > max_val:
                    max_val = current_sum
            dp_current[j] = max_val
        dp_prev = dp_current[:]  # 更新dp_prev为当前段的结果
    
    return dp_prev[n]


def precompute_all_k(n, max_k,cache_result,total_gpu_type_num_dict,hash_list_of_dicts):


    def fun(i,j):
        gpu_type_num_dict = idx_to_gpu_type_num_dict(i-1,j-1,total_gpu_type_num_dict)
        cache_prex = hash_list_of_dicts(gpu_type_num_dict)
        if cache_prex not in cache_result or cache_result[cache_prex] == None:
            return -MAX_VALUE

        return cache_result[cache_prex].thpt

    dp = {}  # dp[m][j] 表示将前j个元素分成m段的最大和
    split_history = {}  # split_history[m][j] 记录分割点

    # 初始化 m=1
    dp[1] = [0] * (n + 1)
    for j in range(1, n + 1):
        dp[1][j] = fun(1, j)
    split_history[1] = [0] * (n + 1)  # m=1时无需分割，但填充占位

    # 动态规划处理 m=2 到 max_k
    for m in range(2, max_k + 1):
        dp_current = [-float('inf')] * (n + 1)
        split_current = [0] * (n + 1)  # 分割点数组长度为n+1
        for j in range(m, n + 1):
            max_val = -float('inf')
            best_i = -1
            for i in range(1, j):
                current_sum = dp[m-1][i] + fun(i, j)
                if current_sum > max_val:
                    max_val = current_sum
                    best_i = i
            dp_current[j] = max_val
            split_current[j] = best_i
        dp[m] = dp_current
        split_history[m] = split_current  # 确保数组长度正确

    return dp, split_history

def get_results_for_ks(n, k_list, dp, split_history):
    results = {}
    for k in k_list:
        if k < 1 or k > len(dp):
            results[k] = (0, [])
            continue
        max_sum = dp[k][n]
        if k == 1:
            results[k] = (max_sum, [])
            continue

        # 回溯分割点：从 m=k 到 m=2
        split_points = []
        current_j = n
        for m in range(k, 1, -1):
            if m not in split_history:
                break
            split_array = split_history[m]
            if current_j < 1 or current_j > n:  # 检查索引合法性
                break
            split_i = split_array[current_j]
            if split_i < 1:  # 分割点至少为1
                break
            split_points.append(split_i)
            current_j = split_i

        split_points.reverse()  # 调整顺序为从左到右
        results[k] = (max_sum, split_points)
    return results


def split_to_cache_result(split, cache_result,total_gpu_type_num_dict,hash_list_of_dicts):
    prev = 0

    config_list = []
    for i in range(len(split)):
        gpu_type_num_dict = idx_to_gpu_type_num_dict(prev,split[i],total_gpu_type_num_dict)
        cache_prex = hash_list_of_dicts(gpu_type_num_dict)
        if cache_prex not in cache_result:
            return []
        prev = split[i]
        config_list.append(cache_result[cache_prex])

    gpu_type_num_dict = idx_to_gpu_type_num_dict(prev, len(total_gpu_type_num_dict),total_gpu_type_num_dict)
    cache_prex = hash_list_of_dicts(gpu_type_num_dict)
    if cache_prex not in cache_result:
        return []
    config_list.append(cache_result[cache_prex])

    return config_list


# def get_node_order_list_each_device_group(gpu_type_num_dict,device_group_list):

from collections import Counter
import copy

def group_into_blocks(lst):
    if not lst:
        return []
    blocks = []
    current = [lst[0]]
    for elem in lst[1:]:
        if elem == current[-1]:
            current.append(elem)
        else:
            blocks.append(current)
            current = [elem]
    blocks.append(current)
    return blocks

def merge_blocks(blocks):
    merged = []
    for block in blocks:
        if not merged:
            merged.append(list(block))
        else:
            last_block = merged[-1]
            if last_block[0] == block[0]:
                last_block.extend(block)
            else:
                merged.append(list(block))
    return merged

def is_valid(merged_blocks):
    seen = set()
    for block in merged_blocks:
        char = block[0]
        if char in seen:
            return False
        seen.add(char)
    return True

def generate_unique_permutations(type_list):
    blocks = group_into_blocks(type_list)
    all_perms = itertools.permutations(blocks)
    seen = set()
    result = []
    for perm in all_perms:
        merged = merge_blocks(perm)
        if is_valid(merged):
            combined = []
            for b in merged:
                combined.extend(b)
            combined_tuple = tuple(combined)
            if combined_tuple not in seen:
                seen.add(combined_tuple)
                result.append(list(combined_tuple))
    return result

def generate_valid_permutations(type_list_, num_list_ ,easy_mode = False):


    if(easy_mode):
        type_list = []
        total_res = []
        for key in type_list_.keys():
            type_list.append(json.dumps(type_list_[key]))
        res = generate_unique_permutations(type_list)
        res_ = []
        for item in res:
            item_ = {}
            idx = 0
            for i in item:
                # print(i)
                item_[f"{idx}"]=json.loads(i)
                idx+=1
            res_.append(item_)
        total_res.append(res_)
        return  total_res
    print(type_list_)
    type_list = []
    for key in type_list_.keys():
        type_list.append(json.dumps(type_list_[key]))
    # print(f"type_list {type_list}")
    # 统计各类型的总数量
    num_list__ = copy.deepcopy(num_list_)
    type_counts = Counter(type_list)
    # 检查总和是否符合
    total_res = []
    for num_list in num_list__ :
        num_list_temp = []
        sum_num_list = 0
        for i in range(len(num_list)):
            sum_num_list += num_list[i]
            if(sum_num_list >= type_list_['0']['GPU_NUM']):
                num_list_temp.append(sum_num_list//type_list_['0']['GPU_NUM'])
                sum_num_list = 0
        num_list = num_list_temp
        total = sum(num_list)
        if total != len(type_list):
            print(f"Error: The sum of num_list {total} does not match the total number of devices {len(type_list)}.")
            return []
        
        allocations = []
        num_regions = len(num_list)
        
        # 使用回溯法生成所有有效的类型分配方案
        def backtrack(region_idx, current_counts, current_allocation):
            if region_idx == num_regions:
                # 检查是否所有类型的数量都恰好用完
                if all(count == 0 for count in current_counts.values()):
                    allocations.append(current_allocation.copy())
                return
            
            required = num_list[region_idx]
            # 遍历所有可能的类型，其剩余数量足够当前区域的需求
            for type_name in list(current_counts.keys()):
                if current_counts[type_name] >= required:
                    new_counts = current_counts.copy()
                    new_counts[type_name] -= required
                    if new_counts[type_name] == 0:
                        del new_counts[type_name]
                    new_allocation = current_allocation.copy()
                    new_allocation.append(type_name)
                    backtrack(region_idx + 1, new_counts, new_allocation)
        
        backtrack(0, type_counts.copy(), [])
        
        # 生成所有可能的排列
        permutations = []
        seen = set()
        for alloc in allocations:
            perm = []
            for i in range(len(alloc)):
                perm += [alloc[i]] * num_list[i]
            # 转换为元组以去重
            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                seen.add(perm_tuple)
                permutations.append(list(perm_tuple))

        res_ = []

        for item in permutations:
            item_ = {}
            idx = 0
            for i in item:
                # print(i)
                item_[f"{idx}"]=json.loads(i)
                idx+=1
            res_.append(item_)
        total_res.append(res_)
    return total_res



def test1():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 1 ,"NODE_NUM": 4},
    "1": {"GPU": "910B3", "GPU_NUM": 1 ,"NODE_NUM": 4}
    }
    print(len(unique_permutations(gpu_type_num_dict_)))  

def test2():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 140},
    "1": {"GPU": "910B3", "GPU_NUM": 8 ,"NODE_NUM": 140}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1

    res=split_devices_to_pipeline_large(gpu_type_num_dict,4)
    # print(res)
    print(len(res))

def test3():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 2}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1
    total_1 = 0
    total_2 = 0
    for i in range(1,17):
        res=split_devices_to_pipeline(gpu_type_num_dict,i)
        total_1+=(len(res))
        # res  =split_devices_to_pipeline_large(gpu_type_num_dict,i)
        # total_2+=(len(res))
        print(res)
    print(total_1)
    print(total_2)

def test3_():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 16}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1
    total_1 = 0
    total_2 = 0
    list_list = []

    for i in range(1,17):
        # res=split_devices_to_pipeline(gpu_type_num_dict,i)
        # total_1+=(len(res))
        res  =split_devices_to_pipeline_large(gpu_type_num_dict,i)
        
        total_2+=(len(res))
        if(res==[]):
            break
        # print(res)
        for split in res:
            list = []
            for pipe in split:
                # print(pipe)
                gpu_num = 0 
                for gpu in pipe:
                    gpu_num+=gpu['GPU_NUM']
                list.append(gpu_num)
            list = sorted(list)
            # print(list)
            if(list not in list_list):
                list_list.append(list)    
    print(total_1)
    print(total_2)
    print(len(list_list))
    # print((list_list))
import random


def test4():
    arr = []
    n = 10
    k = 5
    # for i in range(n+1):
    #     arr.append([random.randint(1, 100) for k in range(n+1)])
    
    for i in range(n+1):
        arr.append([])
        for j in range(n+1):
            arr[i].append(sum(range(i, j+1)))

    res= max_fun_sum_with_splits(n, k, arr)
    print(res)

def test5():
# 使用示例
    n = 10
    max_k = 5
    fun = lambda i, j: sum(range(i, j+1))  # 示例函数，假设fun(i,j)是区间和

    # 预处理所有k=1到max_k
    dp_cache, split_cache = precompute_all_k(n, max_k, fun)

    # 输入k的列表，例如k_list = [1, 3, 5]
    k_list = [1, 3, 5]
    results = get_results_for_ks(n, k_list, dp_cache, split_cache)

    # 输出结果
    for k in k_list:
        print(f"k={k}: Max Sum={results[k][0]}, Splits={results[k][1]}")



def test6():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 4},
    "1": {"GPU": "910B3", "GPU_NUM": 8 ,"NODE_NUM": 4}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1
    res = idx_to_gpu_type_num_dict(1,5, gpu_type_num_dict)
    print(res)

def test7():
    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 4},
    "1": {"GPU": "910B3", "GPU_NUM": 8 ,"NODE_NUM": 4}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1
    res = generate_permutations( gpu_type_num_dict)
    print(len(res))
 
def test8():
    # 示例用法
    type_list = []

    gpu_type_num_dict_ = {
    "0": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 21},
    "1": {"GPU": "910B2", "GPU_NUM": 8 ,"NODE_NUM": 2}
    }
    gpu_type_num_dict ={}
    node_id = 0
    for key in gpu_type_num_dict_.keys():
        for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
            gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
            node_id +=1
            type_list.append(json.dumps({"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}))
    
    print(type_list,len(type_list))
    num_list = [8,8]

    res  = generate_valid_permutations(gpu_type_num_dict_, [num_list])


    print(res , len(res))

def test9():
    type_list = ['A', 'B', 'B', 'C', 'C']
    # 生成结果
    result = generate_unique_permutations(type_list)
    # 打印结果
    for lst in result:
        print(lst)


if __name__ =="__main__":
    test8()