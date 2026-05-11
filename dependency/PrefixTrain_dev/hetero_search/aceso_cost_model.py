# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import math
import os 
from model_ops_info import get_op_spec, get_op_list, get_no_recompute_op_list
from aceso_utils import * 
import sys
sys.path.append("/home/ymj/project/Aceso/hetero_search/pp_simulator")
from pp_simulator.simulator import InterleavedOneFOneBGenerator, OperationExecutor, OneFOneBGenerator, EagerOneFOneBGenerator
from pp_simulator.operations import Config, HyperConfig
# args = parse_args()
import aceso_var

# op_list = get_op_list(aceso_var.args)
# ops_not_recomputed = get_no_recompute_op_list(args)

math_log_2 = {1: int(math.log(1, 2)), 2: int(math.log(2, 2)), 4: int(math.log(4, 2)), 6:3 ,8: int(math.log(8, 2)), 16: int(math.log(16, 2))}#TODO
# global_mbs_index = None

# global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
# global reserved_fwd, reserved_bwd 
# global inter_band, intra_band 
# global global_mbs_index




def get_mbs_index(mbs):
    assert aceso_var.global_mbs_index is not None
    return aceso_var.global_mbs_index[mbs]




# def read_profiled_time_(model_name, model_size, time_path):
#     global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index
    
#     mbs_list = args.micro_batch_size
#     global_mbs_index = {}
#     for i in range(len(mbs_list)):
#         global_mbs_index[mbs_list[i]] = i
#     # print(f"global_mbs_index: {global_mbs_index}")
#     if (model_name == "gpt" and model_size == "350M") or (model_name == "t5" and model_size == "220M"):
#         max_tp_size = min(args.max_tp, 4)
#     else:
#         max_tp_size = min(args.max_tp, 8)

#     tp_size_list = []
#     tp = 1
#     while tp <= max_tp_size:
#         tp_size_list.append(tp)
#         tp *= 2
#     comm_num_gpus_list = tp_size_list[1:]

#     algo_list = [0] if model_name == "t5" else [0, 1]

#     compute_fwd_time = {}
#     compute_bwd_time = {}
#     input_size = {}
#     output_size = {}
#     weights = {}
#     activations = {}
#     reserved_fwd = {}
#     reserved_bwd = {}

#     global op_list                                    

#     ## T5 22B and 11B share same op.
#     if model_name == "t5" and model_size == "22B":
#         model_size = "11B"

#     for gpu_type in args.gpu_type_set:
#         compute_fwd_time[gpu_type] = {}
#         compute_bwd_time[gpu_type] = {}
#         # input_size = {}
#         # output_size = {}
#         # weights = {}
#         # activations = {}
#         # reserved_fwd = {}
#         # reserved_bwd = {}

#         for op_name in op_list:
#             compute_fwd_time[gpu_type][op_name] = []
#             compute_bwd_time[gpu_type][op_name] = []
#             input_size[op_name] = []
#             output_size[op_name] = []
#             weights[op_name] = []
#             activations[op_name] = []

#             reserved_fwd[op_name] = []
#             reserved_bwd[op_name] = []

#             for i in range(len(mbs_list)):
#                 compute_fwd_time[gpu_type][op_name].append([])
#                 compute_bwd_time[gpu_type][op_name].append([])
#                 input_size[op_name].append([])
#                 output_size[op_name].append([])  
#                 weights[op_name].append([])   
#                 activations[op_name].append([])  

#                 reserved_fwd[op_name].append([])  
#                 reserved_bwd[op_name].append([])  

#                 for j in range(len(tp_size_list)):                
#                     compute_fwd_time[gpu_type][op_name][i].append([])
#                     compute_bwd_time[gpu_type][op_name][i].append([])
#                     input_size[op_name][i].append([])
#                     output_size[op_name][i].append([])   
#                     weights[op_name][i].append([])      
#                     activations[op_name][i].append([])   

#                     reserved_fwd[op_name][i].append([])    
#                     reserved_bwd[op_name][i].append([])    

#                     for k in range(len(algo_list)):
#                         compute_fwd_time[gpu_type][op_name][i][j].append(1000000)
#                         compute_bwd_time[gpu_type][op_name][i][j].append(1000000)
#                         input_size[op_name][i][j].append(1000000)
#                         output_size[op_name][i][j].append(1000000)   
#                         weights[op_name][i][j].append(1000000)      
#                         activations[op_name][i][j].append(1000000)   

#                         reserved_fwd[op_name][i][j].append(1000000)  
#                         reserved_bwd[op_name][i][j].append(1000000)                  

#     for gpu_type in args.gpu_type_set:     
#         for mbs in mbs_list:
#             for tp in tp_size_list:
#                 mbs_index = get_mbs_index(mbs)
#                 tp_index = int(math.log(tp, 2))
#                 for algo_index in algo_list:
#                     if model_name == "scale-layer":
#                         src_data_file = time_path + f"gpt_scale-layer_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
#                     else:
#                         # src_data_file = time_path + model_name + f"_{model_size}_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
#                         src_data_file = time_path + f"{gpu_type}/"+f"{model_name}_{model_size}/"+ model_name + f"_{model_size}_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
#                     try:
#                         with open(src_data_file) as f:
#                             src_data = csv.reader(f)
#                             line_index = 0
#                             for row in src_data:
#                                 line_index += 1
#                                 if line_index > 1:
#                                     op_name = row[0]
#                                     compute_fwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[1])
#                                     compute_bwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[2])
#                                     input_size[op_name][mbs_index][tp_index][algo_index] = float(row[3])
#                                     output_size[op_name][mbs_index][tp_index][algo_index] = float(row[4]) 
#                                     weights[op_name][mbs_index][tp_index][algo_index] =  float(row[5])   
#                                     activations[op_name][mbs_index][tp_index][algo_index] =  float(row[6])     

#                                     if args.consider_reserved_space:
#                                         reserved_fwd[op_name][mbs_index][tp_index][algo_index] =  float(row[7])  
#                                         reserved_bwd[op_name][mbs_index][tp_index][algo_index] =  float(row[8])                                              
#                     except: 
#                         continue
#                         # print(f"file ({src_data_file}) not exist, or the file is not formatted as expected.")
#     global collective_time 
#     collective_time = {}
#     if model_name in ["gpt", "scale-layer"]:
#         prim_list = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
#     elif model_name in ["t5"]:
#         prim_list = []
#     elif model_name in ["resnet"]:
#         prim_list = ["all_gather", "all_to_all"]
        
#     for gpu_type in args.gpu_type_set:
#         collective_time[gpu_type] = {}
#         for prim in prim_list:
#             collective_time[gpu_type][prim] = {}
#             for num_gpus in comm_num_gpus_list:
#                 collective_time[gpu_type][prim][num_gpus] = {}
#                 if model_name == "scale-layer":
#                     src_data_file = time_path + f"prim_gpt_scale-layer_{prim}_{num_gpus}gpus.csv"
#                 else:
#                     src_data_file = time_path + gpu_type+ f"/{args.model_name}_{model_size}/"+ f"prim_{model_name}_{model_size}_{prim}_{num_gpus}gpus.csv"
#                     # print(f"src_data_file: {src_data_file}")
#                 if(os.path.exists(src_data_file)): #TODO Q：为什么读取A100的信息后，结果没有变化
#                     with open(src_data_file) as f:
#                         src_data = csv.reader(f)
#                         line_index = 0
#                         for row in src_data:
#                             line_index += 1
#                             if line_index > 1:
#                                 data_size = row[0]
#                                 # print(f"gpu_type: {gpu_type}, prim: {prim}, num_gpus: {num_gpus}, data_size: {data_size}")
#                                 collective_time[gpu_type][prim][num_gpus][data_size] = float(row[1])
#                 else:
#                     print(f"file ({src_data_file}) not exist.")
#     global inter_band, intra_band
#     intra_band = {}
#     inter_band_file = time_path +"p2p_inter_node.csv"
#     for gpu_type in args.gpu_type_set:
#         intra_band_file = time_path + gpu_type + "/p2p_intra_node.csv"
#         try:
#             with open(intra_band_file) as f:
#                 src_data = csv.reader(f)
#                 for idx, row in enumerate(src_data):
#                     if idx == 1:
#                         intra_band[gpu_type] = [float(row[i]) for i in range(len(row))]
#             print(f"intra-node bandwidth {gpu_type} = {intra_band[gpu_type]}")
#         except:
#             print(f"intra-node bandwidth file:{intra_band_file} is not found.")

#     try:
#         with open(inter_band_file) as f:
#             src_data = csv.reader(f)
#             for idx, row in enumerate(src_data):
#                 if idx == 1:
#                     inter_band = [float(row[i]) for i in range(len(row))]
#     except:
#         print(f"inter-node bandwidth file is not found, using intra-node bandwidth instead.")
#         inter_band = intra_band[0] 

#     return (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index)

# def extend_profiled():
#     global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index
#     for gpu_type in args.gpu_type_set:     
#         for mbs in mbs_list:
#             for tp in tp_size_list:
#                 mbs_index = get_mbs_index(mbs)
#                 tp_index = int(math.log(tp, 2))
#                 for algo_index in algo_list:
#                             line_index = 0
#                             for row in src_data:
#                                 line_index += 1
#                                 if line_index > 1:
#                                     op_name = row[0]
#                                     compute_fwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[1])
#                                     compute_bwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[2])
#                                     input_size[op_name][mbs_index][tp_index][algo_index] = float(row[3])
#                                     output_size[op_name][mbs_index][tp_index][algo_index] = float(row[4]) 
#                                     weights[op_name][mbs_index][tp_index][algo_index] =  float(row[5])   
#                                     activations[op_name][mbs_index][tp_index][algo_index] =  float(row[6])     

#                                     if args.consider_reserved_space:
#                                         reserved_fwd[op_name][mbs_index][tp_index][algo_index] =  float(row[7])  
#                                         reserved_bwd[op_name][mbs_index][tp_index][algo_index] =  float(row[8])                                              
#                     except: 
#                         print(f"file ({src_data_file}) not exist, or the file is not formatted as expected.")
#     global collective_time 
#     collective_time = {}
#     if model_name in ["gpt", "scale-layer"]:
#         prim_list = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
#     elif model_name in ["t5"]:
#         prim_list = []
#     elif model_name in ["resnet"]:
#         prim_list = ["all_gather", "all_to_all"]
#     for gpu_type in args.gpu_type_set:
#         collective_time[gpu_type] = {}
#         for prim in prim_list:
#             collective_time[gpu_type][prim] = {}
#             for num_gpus in comm_num_gpus_list:
#                 collective_time[gpu_type][prim][num_gpus] = {}
#                 if model_name == "scale-layer":
#                     src_data_file = time_path + f"prim_gpt_scale-layer_{prim}_{num_gpus}gpus.csv"
#                 else:
#                     src_data_file = time_path + gpu_type+ f"/{args.model_name}_{model_size}/"+ f"prim_{model_name}_{model_size}_{prim}_{num_gpus}gpus.csv"
#                 with open(src_data_file) as f:
#                     src_data = csv.reader(f)
#                     line_index = 0
#                     for row in src_data:
#                         line_index += 1
#                         if line_index > 1:
#                             data_size = row[0]
#                             collective_time[gpu_type][prim][num_gpus][data_size] = float(row[1])

def identical_spec(input_spec, required_spec):
    identical = True 
    if input_spec is None or required_spec is None:
        return identical

    if input_spec["R"] != required_spec["R"]:
        identical = False
    if input_spec["V"] != required_spec["V"]:
        identical = False    
    for dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][dim_index] != required_spec["dims"][dim_index]:
            identical = False
    
    return identical

def get_reshard_primitives(input_spec, required_spec):
    if identical_spec(input_spec, required_spec):
        return None, None, 0

    if input_spec["R"] > required_spec["R"]:
        ## R -> Dim, split
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["R"] % required_spec["R"] == 0
                num_devices = input_spec["R"] // required_spec["R"]

                return "split", "all_gather", num_devices

    elif input_spec["V"] > required_spec["V"]:
        ## V -> R, all-reduce
        if input_spec["R"] < required_spec["R"]:
            assert input_spec["V"] % required_spec["V"] == 0
            num_devices = input_spec["V"] // required_spec["V"]

            return "all_reduce", None, num_devices       

        ## V-> D, reduce-scatter
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["V"] % required_spec["V"] == 0
                num_devices = input_spec["V"] // required_spec["V"]

                return "reduce_scatter", "all_gather", num_devices

    else:
        for src_dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][src_dim_index] > required_spec["dims"][src_dim_index]:
                ## D -> R, all-gather
                if input_spec["R"] < required_spec["R"]:
                    assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0
                    num_devices = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                    return "all_gather", "split", num_devices

                for dst_dim_index in range(len(input_spec["dims"])):
                    ## D -> D, all-to-all
                    if dst_dim_index != src_dim_index and input_spec["dims"][dst_dim_index] < required_spec["dims"][dst_dim_index]:
                        assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0
                        num_devices = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                        return "all_to_all", "all_to_all", num_devices

def get_reshard_time(prim, num_devices, data_size,node_id_num):
    assert num_devices > 1
    max_time = -1
    # assert len(node_id_num.keys())==1 , f"only support one gpu type ,but len(node_id_num.keys()) ={len(node_id_num.keys())}"

    for gpu_type in node_id_num.keys():
        if prim in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
            collective_time_ = get_collective_time(gpu_type,prim, num_devices)
            _data_size = '{:.0f}'.format(float(data_size))
            if _data_size in collective_time_:
                max_time = max(max_time, collective_time_[_data_size])
            elif '{:.0f}'.format(float(data_size) - 1) in collective_time_:
                max_time = max(max_time, collective_time_['{:.0f}'.format(float(data_size) - 1)])
            else:
                return 100000
        elif prim in ["split"]:
            return 0
        else:
            return 100000
    return max_time
    # if prim in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
    #     _data_size = '{:.0f}'.format(float(data_size))
    #     if _data_size in collective_time[gpu_type][prim][num_devices]:
    #         return collective_time[gpu_type][prim][num_devices][_data_size]
    #     elif '{:.0f}'.format(float(data_size) - 1) in collective_time[gpu_type][prim][num_devices]:
    #         return collective_time[gpu_type][prim][num_devices]['{:.0f}'.format(float(data_size) - 1)]
    #     else:
    #         return 100000
    # elif prim in ["split"]:
    #     return 0
    # else:
    #     return 100000

def get_reshard_memory(prim, num_devices, data_size):
    assert num_devices > 1
    if prim == "all_reduce":
        return data_size
    elif prim == "all_gather":
        return data_size * num_devices
    elif prim == "reduce_scatter":
        return data_size
    elif prim == "split":
        return data_size
    elif prim == "all_to_all":
        return data_size * num_devices

def intra_node_band(data_size,node_id_num):
    intra_band = aceso_var.intra_band
    # assert len(node_id_num.keys())==1 , f"only support one gpu type,but len(node_id_num.keys()) ={len(node_id_num.keys())}" #TODO
    gpu_type = list(node_id_num.keys())[0]
    if data_size > 0:
        index =  int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(intra_band): #TODO the drawback of aceso
            return intra_band[gpu_type][-1] * 0.001 
        else:
            return intra_band[gpu_type][index] * 0.001
    else:
        return 1

def inter_node_band(data_size):
    inter_band = aceso_var.inter_band
    if data_size > 0:
        index =  int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(inter_band):
            return inter_band[-1] * 0.001
        else:
            return inter_band[index] * 0.001
    else:
        return 1

def get_comm_time(node_id_num,in_cross_node,out_cross_node,input_comm_size,output_comm_size): #TODO 没考虑下一个stage的GPU分布情况
    max_gpu_num = 0
    for gpu_type in node_id_num.keys():
        max_gpu_num = max(max_gpu_num,node_id_num[gpu_type])
    input_comm_size*=min(max_gpu_num ,8)
    output_comm_size*=min(max_gpu_num ,8)
    if in_cross_node:
        in_comm = input_comm_size/inter_node_band(input_comm_size)
    else:
        in_comm = input_comm_size/intra_node_band(input_comm_size,node_id_num)
    if out_cross_node:
        out_comm = output_comm_size/inter_node_band(output_comm_size)
    else:
        out_comm = output_comm_size/intra_node_band(output_comm_size,node_id_num)
    
    return in_comm, out_comm
    
        

def get_time_v3(node_id_num,ops, mbs, tp, algo, dp, cp , in_cross_node, out_cross_node, first_stage = False ,laset_stage = False ): 
    args = aceso_var.args

    if len(ops) == 0:
        return 0, 0, 0, 0, 0
    compute_fwd_time, compute_bwd_time, input_size, output_size = aceso_var.compute_fwd_time , aceso_var.compute_bwd_time, aceso_var.input_size, aceso_var.output_size
    fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = 0, 0, 0, 0, 0

    # 输出gpu_type的key
    # print(node_id_num.keys())
    
    for i in range(len(ops)):
        op_name = ops[i]
        # print(f"mbs[{i}] {mbs[i]}")
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp[i], 2))    
        cp_index = int(math.log(cp[i], 2))    
        algo_index = algo[i]
        max_fwd_comp = 0
        max_bwd_comp = 0
        for node_id in node_id_num.keys():
            # max_fwd_comp = max(max_fwd_comp, compute_fwd_time[node_id][op_name][mbs_index][tp_index][algo_index])
            # max_bwd_comp = max(max_bwd_comp, compute_bwd_time[node_id][op_name][mbs_index][tp_index][algo_index])
            max_fwd_comp = max(max_fwd_comp, get_profiled_data(compute_fwd_time,op_name,mbs_index,tp_index,cp_index, algo_index,node_id))
            max_bwd_comp = max(max_bwd_comp, get_profiled_data(compute_bwd_time,op_name,mbs_index,tp_index,cp_index, algo_index,node_id))
        fwd_comp += max_fwd_comp
        bwd_comp += max_bwd_comp
        # fwd_comp += compute_fwd_time[list(node_id_num.keys())[0]][op_name][mbs_index][tp_index][algo_index]  
        # bwd_comp += compute_bwd_time[list(node_id_num.keys())[0]][op_name][mbs_index][tp_index][algo_index]  
        if args.support_comm_predict and tp[i]>1:
            for op_name_suffix in ["qkv", "dense", "GEMM", "conv", "downsample"]:
                if op_name_suffix in op_name:
                    # tp_comm += get_reshard_time("all_reduce", tp[i], output_size[op_name][mbs_index][tp_index][algo_index],node_id_num) * 1000
                    tp_comm += get_reshard_time("all_reduce", tp[i], get_profiled_data(output_size,op_name,mbs_index,tp_index, cp_index, algo_index),node_id_num) * 1000

    in_mbs_index = get_mbs_index(mbs[0])
    in_tp_index = int(math.log(tp[0], 2))#TODO
    in_cp_index = int(math.log(cp[0], 2))#TODO
    out_mbs_index = get_mbs_index(mbs[-1])
    out_tp_index = int(math.log(tp[0], 2))
    out_cp_index = int(math.log(cp[0], 2))
    in_algo_index = algo[0]
    out_algo_index = algo[-1]
    # input_comm_size = input_size[ops[0]][in_mbs_index][in_tp_index][in_algo_index]
    # output_comm_size = output_size[ops[-1]][out_mbs_index][out_tp_index][out_algo_index]
    input_comm_size = get_profiled_data(input_size,ops[0],in_mbs_index,in_tp_index,in_cp_index, in_algo_index)
    output_comm_size = get_profiled_data(output_size,ops[-1],out_mbs_index,out_tp_index,out_cp_index, out_algo_index)

    if(args.comm_revised):
        in_comm, out_comm = get_comm_time(node_id_num,in_cross_node,out_cross_node,input_comm_size,output_comm_size)
    else:
        if in_cross_node:
            in_comm = input_comm_size/inter_node_band(input_comm_size)
        else:
            in_comm = input_comm_size/intra_node_band(input_comm_size,node_id_num)

        if out_cross_node:
            out_comm = output_comm_size/inter_node_band(output_comm_size)
        else:
            out_comm = output_comm_size/intra_node_band(output_comm_size,node_id_num)
    if first_stage:
        in_comm =0 
    if laset_stage:
        out_comm =0


    fwd_reshard = 0
    bwd_reshard = 0
    if args.resharding:
        for i in range(1, len(ops)):
            prev_spec = get_op_spec(ops[i-1], tp[i-1], dp[i-1], algo[i-1], input_spec=False)
            current_spec = get_op_spec(ops[i], tp[i], dp[i], algo[i], input_spec=True)
            fwd_prim, bwd_prim, num_devices = get_reshard_primitives(prev_spec, current_spec)
            mbs_index = get_mbs_index(mbs[i])
            tp_index = int(math.log(tp[i], 2))    
            if fwd_prim is not None:
                # fwd_reshard += get_reshard_time(fwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]],node_id_num)
                fwd_reshard += get_reshard_time(fwd_prim, num_devices, get_profiled_data(input_size,ops[i],mbs_index,tp_index,cp_index, algo[i]),node_id_num)
            if bwd_prim is not None:
                # bwd_reshard += get_reshard_time(bwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]],node_id_num)
                bwd_reshard += get_reshard_time(bwd_prim, num_devices, get_profiled_data(input_size,ops[i],mbs_index,tp_index,cp_index, algo[i]),node_id_num)
    # print(f"fwd_res:{fwd_reshard}, bwd_res:{bwd_reshard}")

    in_comm += fwd_reshard * 1000
    out_comm += bwd_reshard * 1000

    tp_comm += fwd_reshard * 1000 + bwd_reshard * 1000
    return fwd_comp, bwd_comp, in_comm, out_comm, tp_comm

def get_recompute_time_v3(gpu_type, ops, recompute_ops, mbs, tp,cp,algo):
    if len(ops) == 0 or sum(recompute_ops) == 0:
        return 0
    compute_fwd_time = aceso_var.compute_fwd_time
    fwd_comp = 0
    
    debug_string = ""
    for i in range(len(ops)):
        if recompute_ops[i] == 1:
            debug_string += ops[i] + ", "
            mbs_index = get_mbs_index(mbs[i])
            tp_index = int(math.log(tp[i], 2))
            cp_index = int(math.log(cp[i], 2))
            algo_index = algo[i]             
            max_fwd_comp = 0
            for node_id in gpu_type.keys():  #TODO 能否支持异构的DP
                # max_fwd_comp = max(max_fwd_comp, compute_fwd_time[node_id][ops[i]][mbs_index][tp_index][algo_index])
                max_fwd_comp = max(max_fwd_comp, get_profiled_data(compute_fwd_time,ops[i],mbs_index,tp_index,cp_index,algo_index,node_id))
            fwd_comp += max_fwd_comp
            # fwd_comp += compute_fwd_time[list(gpu_type.keys())[0]][ops[i]][mbs_index][tp_index][algo_index] 

    return fwd_comp


def get_profiled_data(dict,ops, in_mbs_index, in_tp_index, in_cp_index,algo,gpu_type=None):

    input_size , output_size , weights , activations , reserved_fwd , reserved_bwd = aceso_var.input_size, aceso_var.output_size, aceso_var.weights, aceso_var.activations, aceso_var.reserved_fwd, aceso_var.reserved_bwd

    if (gpu_type is None):
        # max_in_mbs_index = len(dict[ops]) - 1
        max_in_tp_index = len(dict[ops][in_mbs_index]) - 1
        # power_mbs  = 1
        power_tp = 1
        # if(in_mbs_index > max_in_mbs_index):
        #     power_mbs = 2**(in_mbs_index - max_in_mbs_index)
        #     in_mbs_index = max_in_mbs_index
        if(in_tp_index > max_in_tp_index):
            for i in range(in_tp_index - max_in_tp_index ):
                dict[ops][in_mbs_index].append({})
            # dict[ops][in_mbs_index][in_tp_index][algo] = dict[ops][in_mbs_index][max_in_tp_index][algo]/power_tp*1.5 #TODO *1.5
            if(dict == input_size or dict == output_size or dict == weights or dict == activations or dict == reserved_fwd or dict == reserved_bwd):
                dict[ops][in_mbs_index][in_tp_index] = dict[ops][in_mbs_index][max_in_tp_index]
            else:
                dict[ops][in_mbs_index][in_tp_index]= dict[ops][in_mbs_index][max_in_tp_index]
            # in_tp_index = max_in_tp_index

        # if(power_mbs != 1 or power_tp != 1):
        #     print(f"Warning: mbs {in_mbs_index} or tp {in_tp_index} is out of the range max_mbs {len(dict[ops]) - 1} and max_tp {len(dict[ops][in_mbs_index]) - 1} , using the max value instead.")
        try:
            return dict[ops][in_mbs_index][in_tp_index][in_cp_index][algo]
        except Exception as e:

            print(e) 
            print(sys.exc_info())
            print(f"dict {aceso_var.compute_fwd_time['910B3'][ops][in_mbs_index]} ")
            print(f"dict {dict[ops][in_mbs_index]} ops:{ops}, in_mbs_index:{in_mbs_index}, in_tp_index:{in_tp_index},  in_cp_index:{in_cp_index} , algo:{algo} ")
            print(f"dict[ops][in_mbs_index][in_tp_index][algo] {dict[ops][in_mbs_index][in_tp_index][in_cp_index][algo]} ")
            # print(f"gpu_type: {gpu_type}, ops: {ops}, in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, max_in_tp_index {max_in_tp_index}, algo: {algo}")
            # print("dict[ops]",dict[ops],in_mbs_index,in_tp_index)
            # return dict[ops][in_mbs_index][in_tp_index][algo]
            return MAX_VALUE
            # exit()
    else:
        # max_in_mbs_index = len(dict[gpu_type][ops]) - 1
        max_in_tp_index = len(dict[gpu_type][ops][in_mbs_index]) - 1
        # power_mbs  = 1
        power_tp = 1
        # if(in_mbs_index > max_in_mbs_index):
        #     power_mbs = 2**(in_mbs_index - max_in_mbs_index)
        #     in_mbs_index = max_in_mbs_index
        if(in_tp_index > max_in_tp_index):
            for i in range(in_tp_index - max_in_tp_index ):
                dict[gpu_type][ops][in_mbs_index].append({})
            power_tp = 2**(in_tp_index - max_in_tp_index)
            # dict[gpu_type][ops][in_mbs_index][in_tp_index][algo] = dict[gpu_type][ops][in_mbs_index][max_in_tp_index][algo]/power_tp*1.5 #TODO *1.5
            if(dict == input_size or dict == output_size or dict == weights or dict == activations or dict == reserved_fwd or dict == reserved_bwd):
                dict[gpu_type][ops][in_mbs_index][in_tp_index]= dict[gpu_type][ops][in_mbs_index][max_in_tp_index] #TODO
            else:
                dict[gpu_type][ops][in_mbs_index][in_tp_index] = dict[gpu_type][ops][in_mbs_index][max_in_tp_index]

        # if(power_mbs != 1 or power_tp != 1):
        #     print(f"Warning: mbs {in_mbs_index} or tp {in_tp_index} is out of the range max_mbs {max_in_mbs_index} and max_tp {max_in_tp_index} , using the max value instead.")
        try:
            return dict[gpu_type][ops][in_mbs_index][in_tp_index][in_cp_index][algo]
        except Exception as e:
            print(e) 
            print(sys.exc_info()) 
            print(f"gpu_type: {gpu_type}, ops: {ops}, in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, max_in_tp_index {max_in_tp_index}, algo: {algo}")
            # print(f" dict[gpu_type][ops][in_mbs_index][in_tp_index] :{dict[gpu_type][ops][in_mbs_index][in_tp_index]}")
            # print(f" dict[gpu_type][ops] :{dict[gpu_type][ops]}")
            return MAX_VALUE
            return dict[gpu_type][ops][in_mbs_index][in_tp_index][cp_index][algo]
def get_collective_time(gpu_type,prim, num_devices): #TODO 先用最大的num_devices 替代
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}")
    collective_time = aceso_var.collective_time
    # print(f"max_num_devices: {max_num_devices}")
    max_num_devices = 0
    # print(f"collective_time[{gpu_type}][{prim}]: {collective_time[gpu_type][prim]}")
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}")
    for key in collective_time[gpu_type][prim].keys():
        max_num_devices = max(max_num_devices, int(key))
        if(int(key) == num_devices):
            return collective_time[gpu_type][prim][key]
    # print(f"Warning: num_devices {num_devices} is out of the range max_num_devices {max_num_devices} , using the max value instead.")
    collective_time[gpu_type][prim][num_devices] = collective_time[gpu_type][prim][max_num_devices] #TODO
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}, max_num_devices: {max_num_devices}")
    return collective_time[gpu_type][prim][num_devices]
    
        

def get_memory_v3(ops, mbs, tp, cp, algo,debug=False): 
    args = aceso_var.args

    input_size , output_size , weights , activations , reserved_fwd , reserved_bwd = aceso_var.input_size, aceso_var.output_size, aceso_var.weights, aceso_var.activations, aceso_var.reserved_fwd, aceso_var.reserved_bwd
    # print(f"mbs[0] {mbs[0]} ")
    # print(f"tp[0] {tp[0]} ")
    in_mbs_index = get_mbs_index(mbs[0])
    # print(f"in_mbs_index {mbs[0]}")
    # print(f"global_mbs_index {global_mbs_index}")
    # if(tp[0] not in [1,2,4,8]):

    #     return MAX_VALUE ,MAX_VALUE ,MAX_VALUE
    in_tp_index = int(math.log(tp[0], 2))      
    in_cp_index = int(math.log(cp[0], 2))      
    # try:
    # inputs = input_size[ops[0]][in_mbs_index][in_tp_index][algo[0]]
    inputs = get_profiled_data(input_size,ops[0],in_mbs_index,in_tp_index,in_cp_index, algo[0])
    # except:
    #     # 输出是哪个out of range
    #     print(f"len(input_size): {len(input_size)}, ")
    #     print(f" len(input_size[ops[0]]): {input_size[ops[0]]}, ")
    #     print(f" len(input_size[ops[0]][in_mbs_index]): {len(input_size[ops[0]][in_mbs_index])}, ")
    #     print(f" len(input_size[ops[0]][in_mbs_index][in_tp_index]): {len(input_size[ops[0]][in_mbs_index][in_tp_index])}")
    #     print(f"in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, algo[0]: {algo[0]}")

    #     # len(nput_size[ops[0]][in_mbs_index]): {len(input_size[ops[0]][in_mbs_index])}, len(input_size[ops[0]][in_mbs_index][in_tp_index]): {len(input_size[ops[0]][in_mbs_index][in_tp_index])}")
    #     print(f"in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, algo[0]: {algo[0]}")
    _activations = 0  
    _weights = 0     
    for i in range(len(ops)):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp[i], 2))
        cp_index = int(math.log(cp[i], 2))
        algo_index = algo[i]
        if args.consider_shared_space and ops[i] == "enc-attention-dropout":             
            # _activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5      
            _activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index, algo_index) * 1.5     
            if(debug):
                  print(f"get_memopry_v3 ops{i}: {ops[i]},  activations: {activations[ops[i]][mbs_index][tp_index][cp_index][algo_index]*1.5}")
        elif args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"] or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):          
            _activations += 0        
        else:            
            # try:
            _activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index,algo_index)        
            # _activations += activations[ops[i]][mbs_index][tp_index][algo_index] 
            if(debug):
                print(f"get_memopry_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][cp_index][algo_index]}")
            # except:
            #     print(f"len(activations): {len(activations)}, len(ops): {len(ops)}, len(mbs): {len(mbs)}, len(tp): {len(tp)}, len(algo): {len(algo)}")
            #     print(f"ops[i]: {ops[i]}, mbs_index: {mbs_index}, tp_index: {tp_index}, algo_index: {algo_index}")     
        # _weights += weights[ops[i]][mbs_index][tp_index][algo_index]
        _weights += get_profiled_data(weights,ops[i],mbs_index,tp_index,cp_index,algo_index)
    if(debug):
        print(f"get_memopry_v3 activations: {_activations}")
    return _weights, inputs, _activations   

def get_activations_v3(ops, recompute_ops, mbs, tp, cp, algo,debug=False):
    args = aceso_var.args

    if len(ops) <= 1 or sum(recompute_ops) == 0:
        return 0
    if(debug):
        print("recompute_ops: ",recompute_ops)
    activations = aceso_var.activations
    saved_activations = 0
    for i in range(len(ops) - 1):
        if recompute_ops[i] == 1 and  recompute_ops[i+1] == 1:
            mbs_index = get_mbs_index(mbs[i])
            tp_index = int(math.log(tp[i], 2))    
            cp_index = int(math.log(cp[i], 2))    
            algo_index = algo[i]               
            if args.consider_shared_space and ops[i] == "enc-attention-dropout":
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5 #q:为什么是1.5
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index,algo_index) * 1.5
                if(debug):
                    print(f"get_activations_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][cp_index][algo_index]*1.5}")
            elif args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):
                saved_activations += 0
            elif ops[i+1] in ["enc-1st-layernorm"] or "-conv1" in ops[i+1]:
                saved_activations += 0
            else:
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index]
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index,algo_index)
                if(debug):
                    print(f"get_activations_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][cp_index][algo_index]}")
        else:
            if(debug):
                print(f"get_activations_v3 ops{i}: {ops[i]}, recompute_ops[i]: {recompute_ops[i]}, recompute_ops[i+1]: {recompute_ops[i+1]}")
    if(debug):
        print(f"get_activations_v3 saved_activations: {saved_activations}")
    return saved_activations

def get_peak_activations(ops, recompute_ops, mbs, tp, cp, algo):
    args = aceso_var.args

    if len(ops) <= 1 or sum(recompute_ops) == 0:
        return 0

    activations = aceso_var.activations
    saved_activations = 0
    saved_activations_list = [0]
    # 对 recompute_ops 进行遍历，如果两个相邻的 op 都需要重计算，那么计算这两个 op 之间的 saved_activations
    # 如果两个相邻的 op 之间有一个不需要重计算，那么将 saved_activations 加入 saved_activations_list 中

    for i in range(len(ops) - 1):
        if recompute_ops[i] == 1 and  recompute_ops[i+1] == 1:
            mbs_index = get_mbs_index(mbs[i])
            tp_index = int(math.log(tp[i], 2))    
            cp_index = int(math.log(cp[i], 2))    
            algo_index = algo[i]      
            if args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"] or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):
                saved_activations += 0
            elif args.consider_shared_space and ops[i] == "enc-attention-dropout":
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index,algo_index) * 1.5
            else:
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index]
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,cp_index,algo_index)
                if ops[i+1] in ["enc-1st-layernorm"] or "-conv1" in ops[i+1] or i + 1 == len(ops) - 1:
                    saved_activations_list.append(saved_activations)
                    saved_activations = 0
        else:
            if saved_activations > 0:
                saved_activations_list.append(saved_activations)
                saved_activations = 0
    
    return max(saved_activations_list)

def get_reserved_memory(ops, mbs, tp, dp, cp, algo, memory_weights):
    args = aceso_var.args
    reserved_fwd, reserved_bwd = aceso_var.reserved_fwd, aceso_var.reserved_bwd
    input_size , output_size , weights , activations = aceso_var.input_size, aceso_var.output_size, aceso_var.weights, aceso_var.activations
    current_reserved_fwd = 0
    current_reserved_bwd = 0
    for i in range(len(ops) - 1):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp[i], 2))    
        cp_index = int(math.log(cp[i], 2))    
        algo_index = algo[i]   
        # if reserved_fwd[ops[i]][mbs_index][tp_index][algo_index] > current_reserved_fwd:
        if get_profiled_data(reserved_fwd,ops[i],mbs_index,tp_index,cp_index,algo_index) > current_reserved_fwd:
            # current_reserved_fwd = reserved_fwd[ops[i]][mbs_index][tp_index][algo_index]
            current_reserved_fwd = get_profiled_data(reserved_fwd,ops[i],mbs_index,tp_index,cp_index,algo_index)
        # if reserved_bwd[ops[i]][mbs_index][tp_index][algo_index] > current_reserved_bwd:
        if get_profiled_data(reserved_bwd,ops[i],mbs_index,tp_index,cp_index,algo_index) > current_reserved_bwd:
            # current_reserved_bwd = reserved_bwd[ops[i]][mbs_index][tp_index][algo_index]
            current_reserved_bwd = get_profiled_data(reserved_bwd,ops[i],mbs_index,tp_index,cp_index,algo_index)

    max_collective = 0
    if args.consider_collective_memory:
        if args.resharding:
            for i in range(1, len(ops)):
                prev_spec = get_op_spec(ops[i-1], tp[i-1], dp[i-1], algo[i-1], input_spec=False)
                current_spec = get_op_spec(ops[i], tp[i], dp[i], algo[i], input_spec=True)
                fwd_prim, bwd_prim, num_devices = get_reshard_primitives(prev_spec, current_spec)
                mbs_index = get_mbs_index(mbs[i])
                tp_index = int(math.log(tp[i], 2))    
                cp_index = int(math.log(cp[i], 2))    
                if fwd_prim is not None:
                    fwd_collective = get_reshard_memory(fwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][cp_index][algo[i]])
                    if fwd_collective > max_collective:
                        max_collective = fwd_collective
                if bwd_prim is not None:
                    bwd_collective = get_reshard_memory(bwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][cp_index][algo[i]])       
                    if bwd_collective > max_collective:
                        max_collective = bwd_collective 

    if args.memory_pred_type == "MAX":
        return max(current_reserved_fwd + current_reserved_bwd, memory_weights) + max_collective
    elif args.memory_pred_type == "MIN":
        return max(current_reserved_fwd, current_reserved_bwd, memory_weights, max_collective)
    else:
        raise RuntimeError(f"unknown args.memory_pred_type {args.memory_pred_type}")

def get_activation_size(op_name, mbs, tp, cp, algo_index=0):
    args = aceso_var.args

    activations = aceso_var.activations
    mbs_index = get_mbs_index(mbs)
    if(tp>args.max_tp):
        return MAX_VALUE

    tp_index = math_log_2[tp]
    cp_index = math_log_2[cp]
    
    max_tp_index = len(activations[op_name][mbs_index]) - 1
    if tp_index > max_tp_index:
        power_tp = 2**(tp_index - max_tp_index)
        tp_index = max_tp_index
        return activations[op_name][mbs_index][tp_index][cp_index][algo_index] / power_tp


    return activations[op_name][mbs_index][tp_index][cp_index][algo_index]

def predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, algo_list, delta=False, on_the_right=False, decrease=True ,first_stage=False,last_stage=False):
    in_cross_node = False
    out_cross_node = False
    mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]
    #TODO 为什么 in_cross_node 和 out_cross_node 都是 False
    fwd_comp, bwd_comp, in_comm, out_comm, _ = get_time_v3(node_id_num,ops, mbs_list, tp_size, algo_list, dp_size, cp_size, in_cross_node, out_cross_node,first_stage ,last_stage) 
    recomp_time = get_recompute_time_v3(node_id_num,ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list)   
    if not delta:          
        sum_time = fwd_comp + bwd_comp + in_comm + out_comm + recomp_time
    else:
        if on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm + recomp_time
        elif not on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm + recomp_time
        elif on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm + recomp_time
        elif not on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm + recomp_time
    # print(f"fwd_comp: {fwd_comp}, bwd_comp: {bwd_comp}, in_comm: {in_comm}, out_comm: {out_comm}, recomp_time: {recomp_time}")
    return sum_time/1000 

def predict_stage_memory(ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, num_stages_behind, algo_list, breakdown=False,debug =False):
    args = aceso_var.args

    mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]    
    #q:activations 和 saved_activations 有什么区别
    memory_weights, inputs, activations = get_memory_v3(ops, mbs_list, tp_size, cp_size, algo_list,debug)      
    memory_gradients = memory_weights
    memory_main_params = memory_weights * args.memory_main_params
    memory_optimizer = memory_weights * args.memory_optimizer
      
    saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list,debug)    # saved_activations 指通过重计算来节省的激活内存                    
    peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list) 

    if args.consider_reserved_space:
        memory_reserved = get_reserved_memory(ops, mbs_list, tp_size, dp_size, cp_size, algo_list, memory_weights)
    else:
        memory_reserved = 0

    memory_activations = (inputs + activations - saved_activations) * (num_stages_behind) #q:为什么不是num_stages_behind+1
    # if(memory_activations<=0):
        # print(f"inputs: {inputs}, activations: {activations}, saved_activations: {saved_activations}, num_stages_behind: {num_stages_behind},len(ops): {len(ops)}")
    memory_peak = inputs + activations - saved_activations + peak_activations

    memory_weights += memory_main_params #q:为什么要加上memory_main_params , memory_main_params是什么  a:memory_main_params是单精度参数所占的内存
    memory_sum = memory_weights + memory_gradients + memory_optimizer + memory_activations + memory_peak + memory_reserved
    if(debug):
        print(f"inputs: {inputs}, activations: {activations}, saved_activations: {saved_activations},(num_stages_behind): {num_stages_behind}")
        print(f"memory_weights: {memory_weights}, memory_gradients: {memory_gradients}, memory_optimizer: {memory_optimizer}, memory_activations: {memory_activations}, memory_peak: {memory_peak}, memory_reserved: {memory_reserved} memory_sum {memory_sum}")
    if breakdown:
        return memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved
    else:
        return memory_sum

def get_cross_node_list(config,inter_comm_gpu):
    cross_node_list = []
    exist_diff_node_gpu_list = []
    current_gpus = 0
    for i in range(config.num_stages):
        stage = config.stages[i]
        num_gpus = stage.num_gpus
        for inter_gpu in inter_comm_gpu:
            if(current_gpus < inter_gpu and current_gpus + num_gpus > inter_gpu):
                # print(f"current_gpus: {current_gpus}, num_gpus: {num_gpus}, inter_gpu: {inter_gpu}")
                exist_diff_node_gpu_list.append(True)
                break
        if(len(exist_diff_node_gpu_list) < i+1):
            exist_diff_node_gpu_list.append(False)
        current_gpus += num_gpus
    # print(exist_diff_node_gpu_list)

    current_gpus = 0
    for i in range(config.num_stages):
        current_gpus += config.stages[i].num_gpus
        if((exist_diff_node_gpu_list[i] or (i+1 < config.num_stages and exist_diff_node_gpu_list[i+1]))or (current_gpus in inter_comm_gpu)):
            cross_node_list.append(True)
        else:
            cross_node_list.append(False)
    return cross_node_list
        

def predict_time_breakdown(config,print_time=False,args=None, print_memory=False):
    base_batch_size = config.micro_bs
    global_batch_size = config.global_bs
    num_batches = global_batch_size // base_batch_size

    _time_list = []
    memory_list = []
    compute_time_list = []
    efficiency_list = []
    gpu_time_list = []
    breakdown_ideal_time_per_gpu_list = []

    breakdown_pure_comp_time_list = []
    breakdown_pure_eff_loss_time_list = []
    breakdown_pure_recomp_time_list = []

    memory_result_strings = []
    time_result_strings = []

    num_gpus_till_now = 0

    
    out_cross_node = False

    if(args==None):
        print("args is None")
    inter_comm_gpu = []
    for i in range(len(args.num_gpus_per_node_list)):
        if(i):
            inter_comm_gpu.append(args.num_gpus_per_node_list[i] + inter_comm_gpu[i-1])
        else:
            inter_comm_gpu.append(args.num_gpus_per_node_list[i])

    each_stage_time_breakdown = []
    each_stage_memory_breakdown = []
    last_out_comm = 0

    cross_node_list = get_cross_node_list(config,inter_comm_gpu)


    for i in range(config.num_stages):
        # print(f"stage {i}")
        stage = config.stages[i]
        ops = stage.ops
        num_gpus = stage.num_gpus
        tp_size = stage.tp_size
        dp_size = stage.dp_size
        cp_size = stage.cp_size
        algo_list = stage.algo
        recompute_ops = stage.recompute_ops
        num_stages_behind = stage.num_stages_behind  
        mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]

        # in_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0 and num_gpus_till_now > 0

        #用于判断是否需要跨节点通信
        in_cross_node = out_cross_node 

        num_gpus_till_now += num_gpus
        # out_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0 
        # out_cross_node = (num_gpus_till_now in inter_comm_gpu) and ( i == config.num_stages - 1 or num_gpus_till_now+ config.stages[i+1].num_gpus in inter_comm_gpu)
        out_cross_node = cross_node_list[i]

        # gpu_type = {} key:gpu_type, value:gpu_num
        # gpu_type = {}
        # temp_num_gpu_till_now = num_gpus_till_now - num_gpus
        # temp_num_gpus = num_gpus
        # print("stage {} num_gpus_till_now = {}, num_gpus = {}".format(i, num_gpus_till_now, num_gpus))
        # boundary_list = get_boundary_list(args)
        # print(f"boundary_list: {boundary_list}")
        # print(f"args.gpu_type_list: {args.gpu_type_list}")
        # for boundary in range(len(boundary_list)):
        #     if(temp_num_gpu_till_now <= boundary_list[boundary]):
        #         while(temp_num_gpus > 0):
        #             gpu_type_ = args.gpu_type_list[boundary]
        #             if(gpu_type.get("gpu_type_")==None):
        #                 gpu_type[gpu_type_] = min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             else:
        #                 gpu_type[gpu_type_] += min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             temp = min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             temp_num_gpu_till_now += temp
        #             temp_num_gpus -= temp
        #             boundary+=1
        #             if(gpu_type[gpu_type_] == 0):
        #                 #delete key gpu_type_
        #                 del gpu_type[gpu_type_]
        #         break

                

        # print(f"stage {i}: gpu_type: {gpu_type}")

        ## compute actual time of each stage
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = get_time_v3(stage.node_id_num,ops, mbs_list, tp_size, algo_list, dp_size, cp_size, in_cross_node, out_cross_node,first_stage = i==0,laset_stage=i==config.num_stages-1)# Q：不同stage之间的通信是否会重复计算
        # print(f"{i} last_out_comm {last_out_comm} in_comm {in_comm}")
        if(args.comm_revised):
            if(i):
                if(last_out_comm > in_comm):
                    in_comm = last_out_comm
                else:
                    _time_list[i-1]+= (in_comm-last_out_comm)/1000
                    gpu_time_list[i-1] += (in_comm-last_out_comm) * num_gpus
                    each_stage_time_breakdown[i-1][0] += (in_comm-last_out_comm)/1000
                    each_stage_time_breakdown[i-1][5] += in_comm-last_out_comm
            last_out_comm = out_comm
                
        
        
        recomp_time = get_recompute_time_v3(stage.node_id_num,ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list)
        sum_time = (fwd_comp + bwd_comp + in_comm + out_comm + recomp_time) / 1000 #TODO 是否可以将communication和compute的时间进行overlap
        _time_list.append(sum_time)
        compute_time_list.append((fwd_comp + bwd_comp + recomp_time) / 1000)
        gpu_time_list.append(sum_time * num_gpus)
        each_stage_time_breakdown.append([sum_time,fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm])
        # print(f"{i} {sum_time,fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm}")
        if print_time:
            time_result_strings.append("[stage {}], {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(i, fwd_comp/1000*num_batches, (bwd_comp + recomp_time)/1000*num_batches, recomp_time/1000*num_batches, in_comm/1000*num_batches, out_comm/1000*num_batches, tp_comm/1000*num_batches))

        ## compute ideal time of each stage
        # print(f"base_batch_size {base_batch_size}")
        _mbs_list = [base_batch_size for _ in range(len(ops))]
        _tp_size = [1 for _ in range(len(ops))]
        _dp_size = [1 for _ in range(len(ops))]
        _cp_size = [1 for _ in range(len(ops))]
        _fwd_comp, _bwd_comp, _in_comm, _out_comm, _tp_comm = get_time_v3(stage.node_id_num,ops, mbs_list, _tp_size, algo_list, _dp_size, _cp_size, in_cross_node, out_cross_node,first_stage = i==0,laset_stage=i==config.num_stages-1) #该stage不考虑并行的时间
        ideal_time = (_fwd_comp + _bwd_comp + _in_comm + _out_comm) / 1000

        ## calculate time breakdown at sum of GPUs. eff_loss_time指的是并行计算的时间损失
        eff_loss_time = (fwd_comp + bwd_comp) - (_fwd_comp + _bwd_comp) / num_gpus #并行计算的时间损失, 该stage的并行计算时间 - 该stage不考虑并行的时间*gpu数， 后置为理想时间

        ## calculate time breakdown per GPU
        breakdown_ideal_time_per_gpu_list.append(((_fwd_comp + _bwd_comp)/num_gpus)/ 1000)
        breakdown_pure_eff_loss_time_list.append(eff_loss_time / 1000)
        breakdown_pure_recomp_time_list.append(recomp_time / 1000)

        ## compute memory
        memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved = \
            predict_stage_memory(ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, num_stages_behind, algo_list, breakdown=True)
        memory_sum = memory_weights + memory_gradients + memory_optimizer + memory_activations + memory_peak + memory_reserved
        memory_list.append(memory_sum)
        if(abs(memory_sum -144004.124) < 0.1):
            print(f"stage {i} memory_sum = {memory_sum}")
        each_stage_memory_breakdown.append([ memory_sum,memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved])
        if print_memory:
            memory_result_strings.append(f"[stage {i}] memory = {memory_sum:.2f} MB. weights = {memory_weights:.0f}, gradients = {memory_gradients:.0f}, optimizer = {memory_optimizer:.0f}, activations = {memory_activations:.0f}, peak += {memory_peak:.0f}, memory_reserved = {memory_reserved:.0f}")
    
        efficiency_list.append(ideal_time / (sum_time * num_gpus))
    sum_stage_time = sum(_time_list)
    time_list = []
    max_time = 0
    bottleneck = 0
    for i in range(config.num_stages):
        time_stage = (_time_list[i] * (num_batches - 1) + sum_stage_time)
        time_list.append(time_stage)
        if print_time:
            time_result_strings[i] += f"{time_stage:.2f}"
            if time_stage > max_time:
                max_time = time_stage
                bottleneck = i
    if print_time:
        time_result_strings[bottleneck] = " * " + time_result_strings[bottleneck]
        print("overall time = {:.2f} ms".format(max_time))
        print("stage, fwd_comp, bwd_comp+recomp, recomp, in_comm(+reshard), out_comm(+reshard), reshard, sum(us)")
        for i in range(config.num_stages):
            print(time_result_strings[i])

    config.time_list = time_list
    config.memory_list = memory_list
    config.compute_time_list = compute_time_list
    config.total_gpu_time = sum(gpu_time_list) * (num_batches - 1)
    config.breakdown_ideal_time_per_gpu = breakdown_ideal_time_per_gpu_list
    config.breakdown_eff_loss_time_per_gpu = breakdown_pure_eff_loss_time_list
    config.breakdown_recomp_time_per_gpu = breakdown_pure_recomp_time_list
    config.each_stage_memory_breakdown = each_stage_memory_breakdown
    config.each_stage_time_breakdown = each_stage_time_breakdown
    config._time_list = _time_list
    config.sum_stage_time = sum_stage_time
    max_time = max(time_list)
    config.thpt = config.global_bs / (max_time / 1000 )

    # max_mem = args.memory_limit
    efficient_time_list = []
    for i in range(config.num_stages):
        used_time = time_list[i]
        used_memory = memory_list[i]
        device_memory = get_memory_limit(args,config.stages[i].node_id_num)
        idle_time = 0
        if sum(config.stages[i].recompute_ops) > 0:
            idle_time = (max_time - used_time) / 2
        else:
            idle_time_under_max_time = max_time - used_time
            # idle_time_under_max_memory = ((max_mem - used_memory) / used_memory) * used_time
            idle_time_under_max_memory = ((device_memory - used_memory) / used_memory) * used_time
            if idle_time_under_max_memory > idle_time_under_max_time:
                idle_time = idle_time_under_max_time
            else:
                idle_time = idle_time_under_max_memory + (idle_time_under_max_time - idle_time_under_max_memory)/2      
        efficient_time_list.append(idle_time * config.stages[i].num_gpus * efficiency_list[i])
    config.efficient_time_list = efficient_time_list

    if print_memory:
        max_memory = 0
        bottleneck = 0
        for i in range(config.num_stages):
            if (memory_list[i]) > max_memory:
                max_memory = memory_list[i]
                bottleneck = i 
        memory_result_strings[bottleneck] = " * " + memory_result_strings[bottleneck]
        print("\nmax allocated memory = {:.2f} MB".format(max_memory))
        for i in range(config.num_stages):
            print(memory_result_strings[i])    
        print(" ")

    return

def predict_time_breakdown_with_interleave(config,print_time=False,args=None, print_memory=False):
    base_batch_size = config.micro_bs
    global_batch_size = config.global_bs
    num_batches = global_batch_size // base_batch_size

    _time_list = []
    memory_list = []
    compute_time_list = []
    efficiency_list = []
    gpu_time_list = []
    breakdown_ideal_time_per_gpu_list = []

    breakdown_pure_comp_time_list = []
    breakdown_pure_eff_loss_time_list = []
    breakdown_pure_recomp_time_list = []

    memory_result_strings = []
    time_result_strings = []

    num_gpus_till_now = 0

    
    out_cross_node = False

    if(args==None):
        print("args is None")
    inter_comm_gpu = []
    for i in range(len(args.num_gpus_per_node_list)):
        if(i):
            inter_comm_gpu.append(args.num_gpus_per_node_list[i] + inter_comm_gpu[i-1])
        else:
            inter_comm_gpu.append(args.num_gpus_per_node_list[i])

    each_stage_time_breakdown = []
    each_stage_memory_breakdown = []
    last_out_comm = 0

    cross_node_list = get_cross_node_list(config,inter_comm_gpu)
    interleave_fwd_time = []
    interleave_bwd_time = []
    for i in range(config.num_stages):
        # print(f"stage {i}")
        stage = config.stages[i]
        ops = stage.ops
        num_gpus = stage.num_gpus
        tp_size = stage.tp_size
        dp_size = stage.dp_size
        cp_size = stage.cp_size
        algo_list = stage.algo
        recompute_ops = stage.recompute_ops
        num_stages_behind = stage.num_stages_behind  
        mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]

        # in_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0 and num_gpus_till_now > 0

        #用于判断是否需要跨节点通信
        in_cross_node = out_cross_node 

        num_gpus_till_now += num_gpus
        # out_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0 
        # out_cross_node = (num_gpus_till_now in inter_comm_gpu) and ( i == config.num_stages - 1 or num_gpus_till_now+ config.stages[i+1].num_gpus in inter_comm_gpu)
        out_cross_node = cross_node_list[i]

        # gpu_type = {} key:gpu_type, value:gpu_num
        gpu_type = {}
        temp_num_gpu_till_now = num_gpus_till_now - num_gpus
        temp_num_gpus = num_gpus
        # print("stage {} num_gpus_till_now = {}, num_gpus = {}".format(i, num_gpus_till_now, num_gpus))
        boundary_list = get_boundary_list(args)
        # print(f"boundary_list: {boundary_list}")
        for boundary in range(len(boundary_list)):
            if(temp_num_gpu_till_now <= boundary_list[boundary]):
                while(temp_num_gpus > 0):
                    gpu_type_ = args.gpu_type_list[boundary]
                    gpu_type[gpu_type_] = min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
                    temp_num_gpus -= gpu_type[gpu_type_]
                    temp_num_gpu_till_now += gpu_type[gpu_type_]
                    boundary+=1
                    if(gpu_type[gpu_type_] == 0):
                        #delete key gpu_type_
                        del gpu_type[gpu_type_]
                break

                

        # print(f"stage {i}: gpu_type: {gpu_type}")

        ## compute actual time of each stage
        len_ops = len(stage.ops)
        ops_0 = len_ops//args.num_ops_each_layer//2  *args.num_ops_each_layer 
        
        
        if(i == 0 ):
            ops_0 += 1
        # print(f" len_ops {len_ops}  ops_0 {ops_0}")
        fwd_comp_0, bwd_comp_0, in_comm, out_comm_, tp_comm = get_time_v3(gpu_type,ops[:ops_0], mbs_list[:ops_0], tp_size[:ops_0], algo_list[:ops_0], dp_size[:ops_0], cp_size[:ops_0], in_cross_node, out_cross_node,first_stage = i==0,laset_stage=i==config.num_stages-1)# Q：不同stage之间的通信是否会重复计算
        fwd_comp_1, bwd_comp_1, in_comm_, out_comm, tp_comm = get_time_v3(gpu_type,ops[ops_0:], mbs_list[ops_0:], tp_size[ops_0:], algo_list[ops_0:], dp_size[ops_0:], cp_size[ops_0:], in_cross_node, out_cross_node,first_stage = i==0,laset_stage=i==config.num_stages-1)# Q：不同stage之间的通信是否会重复计算
        
        
        # print(f"{i} last_out_comm {last_out_comm} in_comm {in_comm}")
        if(args.comm_revised):
            if(i):
                if(last_out_comm > in_comm):
                    in_comm = last_out_comm
            last_out_comm = out_comm
                
        
        recomp_time_0 = get_recompute_time_v3(gpu_type,ops[:ops_0], recompute_ops[:ops_0], mbs_list[:ops_0], tp_size[:ops_0], cp_size[:ops_0], algo_list[:ops_0])
        recomp_time_1 = get_recompute_time_v3(gpu_type,ops[ops_0:], recompute_ops[ops_0:], mbs_list[ops_0:], tp_size[ops_0:], cp_size[ops_0:], algo_list[ops_0:])

        interleave_fwd_time.append( [fwd_comp_0/1000000,fwd_comp_1/1000000])
        interleave_bwd_time.append([bwd_comp_0/1000000+recomp_time_0/1000000,bwd_comp_1/1000000+recomp_time_1/1000000])




    config.interleave_fwd_time =  interleave_fwd_time
    config.interleave_bwd_time =  interleave_bwd_time



    return


def get_reserved_memory_list(config):
    reserved_mem_list = []
    if config is not None:
        base_batch_size = config.micro_bs
        for i in range(config.num_stages):
            stage = config.stages[i]
            ops = stage.ops
            num_gpus = stage.num_gpus
            tp_size = stage.tp_size
            dp_size = stage.dp_size
            cp_size = stage.cp_size
            algo_list = stage.algo
            recompute_ops = stage.recompute_ops
            num_stages_behind = stage.num_stages_behind  
            mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]

            _, _, _, _, _, reserved_mem = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, num_stages_behind, algo_list, breakdown=True)
            reserved_mem_list.append(reserved_mem)
    return reserved_mem_list

def predict_value_after_move(config, bottleneck, partner, num_ops_moved, metric, inc_gpus=False, dec_gpus=False, dim=None,debug=False):
    base_batch_size = config.micro_bs
    ops = list(config.stages[bottleneck].ops)
    tp_size = list(config.stages[bottleneck].tp_size)
    dp_size = list(config.stages[bottleneck].dp_size)
    cp_size = list(config.stages[bottleneck].cp_size)
    algo_list = list(config.stages[bottleneck].algo)
    recompute_ops = list(config.stages[bottleneck].recompute_ops)
    num_stages_behind = config.stages[bottleneck].num_stages_behind
    num_gpus = config.stages[bottleneck].num_gpus
    node_id_num = config.stages[bottleneck].node_id_num
    if num_ops_moved > 0:
        if bottleneck < partner:
            ops = ops[:-num_ops_moved]
            tp_size = tp_size[:-num_ops_moved]
            dp_size = dp_size[:-num_ops_moved]
            cp_size = cp_size[:-num_ops_moved]
            algo_list = algo_list[:-num_ops_moved]
        else:
            ops = ops[num_ops_moved:]
            tp_size = tp_size[num_ops_moved:]
            dp_size = dp_size[num_ops_moved:]
            cp_size = cp_size[num_ops_moved:]
            algo_list = algo_list[num_ops_moved:]
    
    if inc_gpus:
        if dim == "tp":
            for i in range(len(tp_size)):
                tp_size[i] *= 2
        elif dim == "dp":
            for i in range(len(dp_size)):
                dp_size[i] *= 2

    if dec_gpus:
        if dim == "tp":
            for i in range(len(tp_size)):
                tp_size[i] //= 2
        elif dim == "dp":
            for i in range(len(dp_size)):
                dp_size[i] //= 2            

    if num_ops_moved > 0 or inc_gpus or dec_gpus:
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_check_recompute").start()
        recompute_ops = check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, cp_size, num_stages_behind, algo_list,debug=debug)    
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_check_recompute").stop()
    if metric in ["time", "time_with_efficiency"]:
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_time").start()
        pred_value = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, cp_size,base_batch_size, algo_list,first_stage=bottleneck==0 ,last_stage=bottleneck==config.num_stages-1)
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_time").stop()
    elif metric == "memory":
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_memory").start()
        pred_value = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, cp_size,base_batch_size, num_stages_behind, algo_list,debug=debug)
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_memory").stop()

    else:
        raise RuntimeError(f"metric {metric} not implemented.")
    return  pred_value, recompute_ops

######## recomputation-related functions #########

stage_memory_set = {}
stage_memory_visit = 0
stage_memory_hit = 0

def predict_stage_memory_helper(config, stage_index, ops=None, recompute_ops=None, tp_size=None, dp_size=None, cp_size=None, base_batch_size=None, num_stages_behind=None, algo_list=None):
    global stage_memory_visit, stage_memory_hit, stage_memory_set
    if ops is None:
        ops = config.stages[stage_index].ops
        recompute_ops = config.stages[stage_index].recompute_ops
        tp_size = config.stages[stage_index].tp_size
        dp_size = config.stages[stage_index].dp_size
        base_batch_size = config.stages[stage_index].base_bs
        algo_list = config.stages[stage_index].algo
        num_stages_behind = config.stages[stage_index].num_stages_behind

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}cp{cp_size}rc{recompute_ops}algo{algo_list}bs{base_batch_size}stage{num_stages_behind}"
    stage_memory_visit += 1
    if stage_memory_set.get(config_str) is not None:
        stage_memory_hit += 1
        return stage_memory_set[config_str]

    pred_memory = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, num_stages_behind, algo_list)
    config
    stage_memory_set[config_str] = pred_memory

    return pred_memory 

stage_time_set = {}
stage_time_visit = 0
stage_time_hit = 0

def predict_stage_time_helper(config, stage_index):
    global stage_time_visit, stage_time_hit, stage_time_set
    ops = config.stages[stage_index].ops
    recompute_ops = config.stages[stage_index].recompute_ops
    tp_size = config.stages[stage_index].tp_size
    dp_size = config.stages[stage_index].dp_size
    cp_size = config.stages[stage_index].cp_size
    base_batch_size = config.micro_bs
    algo_list = config.stages[stage_index].algo
    node_id_num = config.stages[stage_index].node_id_num
    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}rc{recompute_ops}algo{algo_list}bs{base_batch_size}"
    stage_time_visit += 1
    if stage_time_set.get(config_str) is not None:
        stage_time_hit += 1
        return stage_time_set[config_str]

    pred_time = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, algo_list,first_stage=stage_index==0 ,last_stage=stage_index==config.num_stages-1)
    stage_time_set[config_str] = pred_time

    return pred_time

## op_groups: {"op_name": {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}}
def get_next_recompute_op_group(ops, recompute_ops, base_batch_size, tp_size, dp_size, cp_size, num_stages_behind, algo_list, op_groups, exceed_memory,debug=False): #TODO ?why TODO
    '''
    从最大的saved activations开始,逐渐减少saved activations,直到内存占用小于阈值
    '''
    max_saved_size = 0
    max_saved_op_name = None
    for op_name in op_groups:
        if not op_groups[op_name]["recomputed"] and op_groups[op_name]["sum_size"] > max_saved_size and op_name not in ["enc-1st-layernorm"] and "-conv1" not in op_name and "-relu" not in op_name and op_name not in get_no_recompute_op_list(aceso_var.args):
            max_saved_size = op_groups[op_name]["sum_size"]
            max_saved_op_name = op_name

    if max_saved_size == 0:
        return None, None
    else:
        if(debug):
            print(f"max_saved_op_name: {max_saved_op_name}, max_saved_size: {max_saved_size}")
        op_groups[max_saved_op_name]["recomputed"] = True
        next_op_index = op_groups[max_saved_op_name]["index"][0] + 1
        if next_op_index >= len(ops):
            if(debug):
                print(f"next_op_index: {next_op_index}, len(ops): {len(ops)} , max_saved_op_name: {max_saved_op_name}")
            return None, None 
        next_op_name = ops[next_op_index]
        op_groups[next_op_name]["recomputed"] = True

        if max_saved_size < exceed_memory:
            return op_groups[max_saved_op_name]["index"], op_groups[max_saved_op_name]["activation_size"]
        else:
            saved_size = 0
            index = 0
            while saved_size <= exceed_memory and index < len(op_groups[max_saved_op_name]["activation_size"]): #TODO 由于只有当前stage和下个stage的recom 为1才会使用重计算
                saved_size += op_groups[max_saved_op_name]["activation_size"][index]
                index += 1
            return list(op_groups[max_saved_op_name]["index"][:index]), list(op_groups[max_saved_op_name]["activation_size"][:index])

def check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, cp_size, num_stages_behind, algo_list,debug=False):
    args = aceso_var.args

    if(args.no_recomp):
        return [0 for _ in range(len(ops))]
    num_ops = len(ops)
    if not args.flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
        return recompute_ops
    recompute_ops = [0 for _ in range(num_ops)]
    # 预测该stage的内存占用
    stage_memory = predict_stage_memory_helper(None, None, ops, recompute_ops, tp_size, dp_size, cp_size, base_batch_size, num_stages_behind, algo_list)
    stage_memory += args.peak_mem_in_backward # args.peak_mem_in_backward = 0

    # 如果内存占用小于阈值，直接全部recompute
    # if stage_memory - args.memory_limit <= 0:
    #     return recompute_ops
    min_memory = get_memory_limit(args,node_id_num)
    # if stage_memory - min_memory <= 0: #TODO 内存够的时候不要重计算
    #     return recompute_ops

    ## generate op groups
    op_groups = {}
    
    for index in range(len(ops)):
        if ops[index] not in op_groups:
            op_groups[ops[index]] = {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}
        op_groups[ops[index]]["index"].append(index)
        tmp_activation_size = get_activation_size(ops[index], base_batch_size//dp_size[index], tp_size[index], cp_size[index], algo_list[index]) * (num_stages_behind + 1)
        op_groups[ops[index]]["activation_size"].append(tmp_activation_size)
        op_groups[ops[index]]["sum_size"] += tmp_activation_size    
    if(debug):
        print(f"op_groups: {op_groups}")
    mbs_list = [base_batch_size // dp_size[i] for i in range(len(dp_size))]

    initial_saved_activations = 0
    initial_peak_activations = 0
    
    #获得当前的saved activations和peak activations
    current_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list)    
    # 从最大的saved activations开始，逐渐减少saved activations，直到内存占用小于阈值

    if(debug):
        print(f"exceed_memory: {stage_memory - min_memory}")
    if(stage_memory - min_memory > 0):
        while stage_memory - min_memory > 0:  #TODO 应该考虑计算时间和内存
            saved_index_group, saved_size_list = get_next_recompute_op_group(ops, recompute_ops, base_batch_size, tp_size, dp_size, cp_size, num_stages_behind, algo_list, op_groups, stage_memory - min_memory,debug=debug)
            if saved_index_group is not None:
                for i in range(len(saved_index_group)):
                    saved_index = saved_index_group[i]  
                    if saved_index >= 0:
                        recompute_ops[saved_index] = 1
                        if saved_index < len(ops) - 1:
                            recompute_ops[saved_index + 1] = 1     
                            if(debug):
                                print(f"saved_index: {saved_index}")
                    else:
                        break                      
            else:
                # if(debug):
                break 
            new_saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list)    
            stage_memory -= (new_saved_activations - initial_saved_activations) * (num_stages_behind + 1)
            initial_saved_activations = new_saved_activations
            new_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list)
            stage_memory += (new_peak_activations - current_peak_activations)
            current_peak_activations = new_peak_activations
    else:
        recompute_ops = get_unsaved_ops(ops, recompute_ops, mbs_list, tp_size, cp_size, algo_list, num_stages_behind,min_memory-stage_memory)

    if(debug):
        print(f"new_saved_activations: {new_saved_activations}, stage_memory: {stage_memory} , min_memory {min_memory}")
    return recompute_ops   




# def check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list,debug=False):
#     num_ops = len(ops)
#     if not args.flex_recompute:
#         recompute_ops = [1 for _ in range(num_ops)]
#         return recompute_ops
#     recompute_ops = [0 for _ in range(num_ops)]
#     # 预测该stage的内存占用
#     stage_memory = predict_stage_memory_helper(None, None, ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list)
#     stage_memory += args.peak_mem_in_backward # args.peak_mem_in_backward = 0
#     min_memory = get_memory_limit(args,node_id_num)

#     op_groups = {}
    
#     for index in range(len(ops)):
#         if ops[index] not in op_groups:
#             op_groups[ops[index]] = {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}
#         op_groups[ops[index]]["index"].append(index)
#         tmp_activation_size = get_activation_size(ops[index], base_batch_size//dp_size[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
#         op_groups[ops[index]]["activation_size"].append(tmp_activation_size)
#         op_groups[ops[index]]["sum_size"] += tmp_activation_size    
#     if(debug):
#         print(f"op_groups: {op_groups}")
#     mbs_list = [base_batch_size // dp_size[i] for i in range(len(dp_size))]

#     initial_saved_activations = 0
#     initial_peak_activations = 0
    
#     #获得当前的saved activations和peak activations
#     current_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)    
#     # 从最大的saved activations开始，逐渐减少saved activations，直到内存占用小于阈值

#     if(debug):
#         print(f"exceed_memory: {stage_memory - min_memory}")
#     if(stage_memory - min_memory > 0):
#         while stage_memory - min_memory > 0:  #TODO 应该考虑计算时间和内存
#             recompute_ops , end= get_saved_ops(ops, recompute_ops, mbs_list, tp_size, algo_list, num_stages_behind,stage_memory - min_memory)
#             new_saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, algo_list)    
#             stage_memory -= (new_saved_activations - initial_saved_activations) * (num_stages_behind + 1)
#             initial_saved_activations = new_saved_activations
#             new_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)
#             stage_memory += (new_peak_activations - current_peak_activations)
#             current_peak_activations = new_peak_activations
#             if(end):
#                 break
#     else:
#         recompute_ops = get_unsaved_ops(ops, recompute_ops, mbs_list, tp_size, algo_list, num_stages_behind,min_memory-stage_memory)
#     return recompute_ops   



def update_recompute(config, stage_idx=None):
    '''
    更新stage的recompute_ops
    '''
    args = aceso_var.args

    if(args.no_recomp):
        # print(args)
        return
    if stage_idx is not None:
        updated_stages = [stage_idx]
    else:
        updated_stages = [i for i in range(config.num_stages)]

    for index in updated_stages:
        stage = config.stages[index]
        stage.recompute_ops = check_recompute(stage.node_id_num, stage.ops, config.micro_bs,  #返回在内存限制下的recompute_ops
            stage.tp_size, stage.dp_size, stage.cp_size, stage.num_stages_behind, stage.algo)

def wrap_predict_delta_time(config, longest_stage, shortest_stage, num_ops_moved, decrease=True):
    base_batch_size = config.micro_bs
    ops = config.stages[longest_stage].ops
    recompute_ops = config.stages[longest_stage].recompute_ops
    tp_size = config.stages[longest_stage].tp_size
    dp_size = config.stages[longest_stage].dp_size
    algo_list = config.stages[longest_stage].algo
    node_id_num = config.stages[longest_stage].node_id_num
    num_ops = len(ops)
    if longest_stage < shortest_stage and decrease:
        ops = ops[num_ops - num_ops_moved:]
        recompute_ops = list(recompute_ops[num_ops - num_ops_moved:])
        tp_size = tp_size[num_ops - num_ops_moved:]
        dp_size = dp_size[num_ops - num_ops_moved:]
        algo_list = algo_list[num_ops - num_ops_moved:]
    elif longest_stage > shortest_stage and decrease:
        ops = ops[0:num_ops_moved]
        recompute_ops = list(recompute_ops[0:num_ops_moved])
        tp_size = tp_size[0:num_ops_moved]
        dp_size = dp_size[0:num_ops_moved]
        algo_list = algo_list[0:num_ops_moved]
    elif longest_stage < shortest_stage and not decrease:
        ops = ops[num_ops - num_ops_moved:]
        recompute_ops = [0 for _ in range(num_ops_moved)]
        tp_size = [config.stages[longest_stage+1].tp_size[0] for _ in range(num_ops_moved) ]
        dp_size = [config.stages[longest_stage+1].dp_size[0] for _ in range(num_ops_moved) ]
        algo_list = algo_list[num_ops - num_ops_moved:]
    elif longest_stage > shortest_stage and not decrease:
        ops = ops[0:num_ops_moved]
        recompute_ops = [0 for _ in range(num_ops_moved)]
        tp_size = [config.stages[longest_stage-1].tp_size[-1] for _ in range(num_ops_moved) ]
        dp_size = [config.stages[longest_stage-1].dp_size[-1] for _ in range(num_ops_moved) ]
        algo_list = algo_list[0:num_ops_moved]
    else:
        raise RuntimeError("")

    pred_delta_time = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list, delta=True, on_the_right= longest_stage < shortest_stage, decrease=decrease,first_stage=longest_stage==0 ,last_stage=longest_stage==config.num_stages-1)
    return pred_delta_time   



def get_unsaved_ops(ops, recompute_ops,mbs,tp_size,cp_size, algo_list, num_stages_behind, unsave_memory ,debug=False):
    args = aceso_var.args

    sort_recom_op_name = aceso_var.sort_recom_op_name
    current_unsaved_memory = 0
    mbs_index = get_mbs_index(mbs[0])
    tp_index = int(math.log(tp_size[0], 2))    
    cp_index = int(math.log(cp_size[0], 2))    
    recom_op_name_sort = aceso_var.get_recom_op_name_sort(args.gpu_type_set[0],mbs_index,tp_index,cp_index,algo_list[0])   #TODO 不能都是0

    for recom_op_name in reversed(recom_op_name_sort):
        for index in range(len(ops)-1):
            if ops[index] == recom_op_name:
                if recompute_ops[index] == 1 and recompute_ops[index+1] == 1:
                    if args.consider_shared_space and ops[index] == "enc-attention-dropout":
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], cp_size[index], algo_list[index]) * (num_stages_behind + 1) * 1.5
                    elif args.consider_shared_space and (ops[index] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[index] or ("-downsample" in ops[index] and "0-0" not in ops[index])):
                        activations_size = 0
                    elif ops[index+1] in ["enc-1st-layernorm"] or "-conv1" in ops[index+1]:
                        activations_size = 0
                    else:
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], cp_size[index] , algo_list[index]) * (num_stages_behind + 1)
                    if current_unsaved_memory + activations_size <= unsave_memory:
                        current_unsaved_memory += activations_size
                        recompute_ops[index] = 0
                        if(index+2>len(ops)-1 or recompute_ops[index+2]==0):
                            recompute_ops[index+1]=0
                            
    return recompute_ops
                   

def get_saved_ops(ops, recompute_ops,mbs,tp_size,cp_size, algo_list, num_stages_behind, save_memory ,debug=False):
    args = aceso_var.args

    sort_recom_op_name = aceso_var.sort_recom_op_name
    current_saved_memory = 0
    mbs_index = get_mbs_index(mbs[0])
    tp_index = int(math.log(tp_size[0], 2))    
    cp_index = int(math.log(cp_size[0], 2))    
    recom_op_name_sort = aceso_var.get_recom_op_name_sort(args.gpu_type_set[0],mbs_index,tp_index, cp_index, algo_list[0])   #TODO 不能都是0

    for recom_op_name in recom_op_name_sort:
        for index in range(len(ops)-1):
            if ops[index] == recom_op_name:
                if recompute_ops[index] == 0 and index != len(ops)-1:
                    if args.consider_shared_space and ops[index] == "enc-attention-dropout":
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1) * 1.5
                    elif args.consider_shared_space and (ops[index] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[index] or ("-downsample" in ops[index] and "0-0" not in ops[index])):
                        activations_size = 0
                    elif ops[index+1] in ["enc-1st-layernorm"] or "-conv1" in ops[index+1]:
                        activations_size = 0
                    else:
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
                    recompute_ops[index] = 1
                    recompute_ops[index+1] = 1 
                    current_saved_memory += activations_size

                    if current_saved_memory  >= save_memory:
                        return recompute_ops, False
    
    return recompute_ops,True


def simulator_config(config):
    args = aceso_var.args

    fwd_per_stage=[]
    bwd_per_stage=[]
    stage_idx =0
    num_stage = config.num_stages
    comm_table=[[ 0 for __ in range(num_stage) ] for _ in range(num_stage)]
    r=[[ True for __ in range(num_stage) ] for _ in range(num_stage)]
    # print("config.each_stage_time_breakdown")
    for time_ in config.each_stage_time_breakdown:
        # print(f"time_ {time_}")
        fwd_per_stage.append(time_[1]/1000)
        bwd_per_stage.append((time_[2]+time_[3])/1000) 
        if(stage_idx<num_stage-1):
            comm_table[stage_idx][stage_idx+1] = time_[5]/1000/1000
            comm_table[stage_idx+1][stage_idx] = time_[5]/1000/1000
        stage_idx+=1


    config_ = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=comm_table, r=r, p=num_stage, m=config.global_bs//config.micro_bs, v=1, c=0, overlap_c=True)
    og = OneFOneBGenerator(config_)
    operations = og.generate()
    executor = OperationExecutor(config_, operations)
    executor.execute()
    ans = executor.makespan()
    print("OneFOneBGenerator simulator time " , ans)

    predict_time_breakdown_with_interleave(config,args=args)


    fwd_per_stage_ = [ [] for _ in range(len(fwd_per_stage))]
    bwd_per_stage_ = [ [] for _ in range(len(fwd_per_stage))]
    fwd_per_stage_ = config.interleave_fwd_time 
    bwd_per_stage_ = config.interleave_bwd_time 
    # print(fwd_per_stage_)
    # print(fwd_per_stage)
    # print(bwd_per_stage_)
    # print(bwd_per_stage)
    # print(comm_table)
    # for i in range(len(fwd_per_stage)):
    #     fwd_per_stage_[i].append(fwd_per_stage[i]/2)
    #     fwd_per_stage_[i].append(fwd_per_stage[i]/2)

    # for i in range(len(bwd_per_stage)):
    #     bwd_per_stage_[i].append(bwd_per_stage[i]/2)
    #     bwd_per_stage_[i].append(bwd_per_stage[i]/2)

    config_ = HyperConfig(f=fwd_per_stage_, b=bwd_per_stage_, a=comm_table, r=r, p=num_stage, m=config.global_bs//config.micro_bs, v=2, c=0, overlap_c=True)
    og = InterleavedOneFOneBGenerator(config_)
    operations = og.generate()
    executor = OperationExecutor(config_, operations)
    executor.execute()
    ans = executor.makespan()
    print("InterleavedOneFOneBGenerator simulator time " ,ans ) #为什么这个需要/1000

################################


if __name__ == "__main__":


    args = aceso_var.args
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)
    if("pipe" in config_dict):
        args.num_gpus = 0
        args.node_order =[]
        args.num_nodes_list = []
        for pipe in config_dict["pipe"]:
            args.node_order.append(pipe["node_info"]  )
            args.num_nodes_list.append(len(pipe["node_info"].keys()))
        
            for key in pipe["node_info"].keys():
                args.num_gpus+=pipe["node_info"][key]["GPU_NUM"]
        config = read_hybrid_config_from_json(args)

        args.gpu_type_list = []
        for pipe_idx  in range(len(config.pipelines)):
            for gpu in args.gpu_type_list_pipe[pipe_idx]:
                args.gpu_type_list.append(gpu)
        args.gpu_type_set =set(args.gpu_type_list)

        aceso_var.read_profiled_time(config_dict["model_name"], config_dict["model_size"], args.profiled_time_path,args)

        print("args.num_gpus_per_node_list_pipe",args.num_gpus_per_node_list_pipe)

        if(config_dict.get("node_info")!=None):
            args.node_order = config_dict["node_info"]
        for pipe_idx  in range(len(config.pipelines)):
            args.global_batch_size = config.pipelines[pipe_idx].global_bs
            pipe_config =  config.pipelines[pipe_idx]
            args.num_nodes = args.num_nodes_list[pipe_idx]
            args.num_gpus_per_node_list = args.num_gpus_per_node_list_pipe[pipe_idx]
            args.gpu_type_list = args.gpu_type_list_pipe[pipe_idx]
            print("args.num_gpus_per_node_list",args.num_gpus_per_node_list)
            print("\n\n\n","==="*10,f"Pipeline {pipe_idx}","==="*10)
            predict_time_breakdown(pipe_config, print_time=0, print_memory=0,args=args)
            save_and_print_configs(pipe_config, args)
            simulator_config(pipe_config)
            pipe_idx += 1
    else:
        
        if(args.config_node_order_idx != None):
            args.node_order_list = generate_permutations(args.gpu_type_num_dict)
            args.node_order = args.node_order_list[args.config_node_order_idx]
        else:
            args.node_order = args.gpu_type_num_dict
        config, config_dict = read_config_from_json(args, return_config_dict=True)
        aceso_var.read_profiled_time(config_dict["model_name"], config_dict["model_size"], args.profiled_time_path,args)

        print("\n\n\n","==="*10,"Results","==="*10)
        predict_time_breakdown(config, print_time=0, print_memory=0,args=args)
        # for stage in config.stages:
        #     recomp=check_recompute(stage.node_id_num,stage.ops, config.micro_bs, stage.tp_size, stage.dp_size, stage.num_stages_behind, stage.algo)
        #     if(recomp!=stage.recompute_ops):
        #         print(f"stage {stage.node_id_num} recomp: {recomp} stage.recompute_ops: {stage.recompute_ops}")

        save_and_print_configs(config, args)
        simulator_config(config)
