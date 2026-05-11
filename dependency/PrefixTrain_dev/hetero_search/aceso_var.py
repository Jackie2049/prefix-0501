from aceso_utils import parse_args 
from model_ops_info import get_op_spec, get_op_list, get_no_recompute_op_list
import math
import os
import csv
import sys


args = parse_args()
global sort_recom_op
global sort_recom_op_name 
global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
global reserved_fwd, reserved_bwd 
global inter_band, intra_band 
global global_mbs_index
global op_list



def get_mbs_index(mbs):
    global global_mbs_index
    assert global_mbs_index is not None
    return global_mbs_index[mbs]


def set_profiled_time(profile_time):
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index,op_list ,collective_time,inter_band, intra_band 
    
    (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band )= profile_time


def read_profiled_time(model_name, model_size, time_path,args):
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index
    global op_list
    
    op_list = get_op_list(args)

    mbs_list = args.micro_batch_size
    global_mbs_index = {}
    for i in range(len(mbs_list)):
        global_mbs_index[mbs_list[i]] = i
    # print(f"global_mbs_index: {global_mbs_index}")
    if (model_name == "gpt" and model_size == "350M") or (model_name == "t5" and model_size == "220M"):
        max_tp_size = min(args.max_tp, 4)
        max_cp_size = min(args.max_cp, 4)
    else:
        max_tp_size = min(args.max_tp, 8)
        max_cp_size = min(args.max_cp, 8)

    tp_size_list = []
    tp = 1
    while tp <= max_tp_size:
        tp_size_list.append(tp)
        tp *= 2
    comm_num_gpus_list = tp_size_list[1:]

    cp_size_list = []
    cp = 1
    while cp <= max_cp_size:
        cp_size_list.append(cp)
        cp *= 2

    algo_list = [0] if model_name == "t5" else [0, 1]

    compute_fwd_time = {}
    compute_bwd_time = {}
    input_size = {}
    output_size = {}
    weights = {}
    activations = {}
    reserved_fwd = {}
    reserved_bwd = {}


    ## T5 22B and 11B share same op.
    if model_name == "t5" and model_size == "22B":
        model_size = "11B"

    for gpu_type in args.gpu_type_set:
        compute_fwd_time[gpu_type] = {}
        compute_bwd_time[gpu_type] = {}
        # input_size = {}
        # output_size = {}
        # weights = {}
        # activations = {}
        # reserved_fwd = {}
        # reserved_bwd = {}

        for op_name in op_list:
            compute_fwd_time[gpu_type][op_name] = []
            compute_bwd_time[gpu_type][op_name] = []
            input_size[op_name] = []
            output_size[op_name] = []
            weights[op_name] = []
            activations[op_name] = []

            reserved_fwd[op_name] = []
            reserved_bwd[op_name] = []

            for i in range(len(mbs_list)):
                compute_fwd_time[gpu_type][op_name].append([])
                compute_bwd_time[gpu_type][op_name].append([])
                input_size[op_name].append([])
                output_size[op_name].append([])  
                weights[op_name].append([])   
                activations[op_name].append([])  

                reserved_fwd[op_name].append([])  
                reserved_bwd[op_name].append([])  

                for j in range(len(tp_size_list)):                
                    compute_fwd_time[gpu_type][op_name][i].append([])
                    compute_bwd_time[gpu_type][op_name][i].append([])
                    input_size[op_name][i].append([])
                    output_size[op_name][i].append([])   
                    weights[op_name][i].append([])      
                    activations[op_name][i].append([])   

                    reserved_fwd[op_name][i].append([])    
                    reserved_bwd[op_name][i].append([])    
                    for c in range(len(cp_size_list)):
                        compute_fwd_time[gpu_type][op_name][i][j].append([])
                        compute_bwd_time[gpu_type][op_name][i][j].append([])
                        input_size[op_name][i][j].append([])
                        output_size[op_name][i][j].append([])   
                        weights[op_name][i][j].append([])      
                        activations[op_name][i][j].append([])   

                        reserved_fwd[op_name][i][j].append([])    
                        reserved_bwd[op_name][i][j].append([]) 
                        for k in range(len(algo_list)):
                            compute_fwd_time[gpu_type][op_name][i][j][c].append(1000000)
                            compute_bwd_time[gpu_type][op_name][i][j][c].append(1000000)
                            input_size[op_name][i][j][c].append(1000000)
                            output_size[op_name][i][j][c].append(1000000)   
                            weights[op_name][i][j][c].append(1000000)      
                            activations[op_name][i][j][c].append(1000000)   

                            reserved_fwd[op_name][i][j][c].append(1000000)  
                            reserved_bwd[op_name][i][j][c].append(1000000)                  

    for gpu_type in args.gpu_type_set:     
        for mbs in mbs_list:
            for tp in tp_size_list:
                mbs_index = get_mbs_index(mbs)
                tp_index = int(math.log(tp, 2))
                for cp in cp_size_list:
                    cp_index = int(math.log(cp, 2))
                    for algo_index in algo_list:
                        if model_name == "scale-layer":
                            src_data_file = time_path + f"{gpu_type}/"+f"{model_name}_{model_size}/"+ model_name + f"_{model_size}_seq{args.seq_len}_mbs{mbs}_tp{tp}_cp{cp}_algo{algo_index}.csv"
                        else:
                            # src_data_file = time_path + model_name + f"_{model_size}_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
                            src_data_file = time_path + f"{gpu_type}/"+f"{model_name}_{model_size}/"+ model_name + f"_{model_size}_seq{args.seq_len}_mbs{mbs}_tp{tp}_cp{cp}_algo{algo_index}.csv"
                        try:
                            with open(src_data_file) as f:
                                src_data = csv.reader(f)
                                line_index = 0
                                for row in src_data:
                                    line_index += 1
                                    if line_index > 1:
                                        op_name = row[0]
                                        compute_fwd_time[gpu_type][op_name][mbs_index][tp_index][cp_index][algo_index] = float(row[1])
                                        compute_bwd_time[gpu_type][op_name][mbs_index][tp_index][cp_index][algo_index] = float(row[2])
                                        input_size[op_name][mbs_index][tp_index][cp_index][algo_index] = float(row[3])
                                        output_size[op_name][mbs_index][tp_index][cp_index][algo_index] = float(row[4]) 
                                        weights[op_name][mbs_index][tp_index][cp_index][algo_index] =  float(row[5])   
                                        activations[op_name][mbs_index][tp_index][cp_index][algo_index] =  float(row[6])     

                                        if args.consider_reserved_space:
                                            reserved_fwd[op_name][mbs_index][tp_index][cp_index][algo_index] =  float(row[7])  
                                            reserved_bwd[op_name][mbs_index][tp_index][cp_index][algo_index] =  float(row[8])                                              
                        except Exception as e: 
                            continue
                            # print(f"file ({src_data_file}) not exist, or the file is not formatted as expected.")

    global collective_time 
    collective_time = {}
    if model_name in ["gpt", "scale-layer"]:
        prim_list = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
    elif model_name in ["t5"]:
        prim_list = []
    elif model_name in ["resnet"]:
        prim_list = ["all_gather", "all_to_all"]
        
    for gpu_type in args.gpu_type_set:
        collective_time[gpu_type] = {}
        for prim in prim_list:
            collective_time[gpu_type][prim] = {}
            for num_gpus in comm_num_gpus_list:
                collective_time[gpu_type][prim][num_gpus] = {}
                if model_name == "scale-layer":
                    src_data_file = time_path + f"prim_gpt_scale-layer_{prim}_{num_gpus}gpus.csv"
                else:
                    src_data_file = time_path + gpu_type+ f"/{args.model_name}_{model_size}/"+ f"prim_{model_name}_{model_size}_{prim}_{num_gpus}gpus.csv"
                    # print(f"src_data_file: {src_data_file}")
                if(os.path.exists(src_data_file)): #TODO Q：为什么读取A100的信息后，结果没有变化
                    with open(src_data_file) as f:
                        src_data = csv.reader(f)
                        line_index = 0
                        for row in src_data:
                            line_index += 1
                            if line_index > 1:
                                data_size = row[0]
                                # print(f"gpu_type: {gpu_type}, prim: {prim}, num_gpus: {num_gpus}, data_size: {data_size}")
                                collective_time[gpu_type][prim][num_gpus][data_size] = float(row[1])
                else:
                    print(f"file ({src_data_file}) not exist.")
    global inter_band, intra_band
    intra_band = {}
    inter_band_file = time_path +"p2p_inter_node.csv"
    for gpu_type in args.gpu_type_set:
        intra_band_file = time_path + gpu_type + "/p2p_intra_node.csv"
        try:
            with open(intra_band_file) as f:
                src_data = csv.reader(f)
                for idx, row in enumerate(src_data):
                    if idx == 1:
                        intra_band[gpu_type] = [float(row[i]) for i in range(len(row))]
            # print(f"intra-node bandwidth {gpu_type} = {intra_band[gpu_type]}")
        except:
            print(f"intra-node bandwidth file:{intra_band_file} is not found.")

    try:
        with open(inter_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    inter_band = [float(row[i]) for i in range(len(row))]
    except:
        print(f"inter-node bandwidth file is not found, using intra-node bandwidth instead.")
        inter_band = intra_band[0] 

    return (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band)





def get_sort_op_by_size_comp(args):
    global sort_recom_op
    global sort_recom_op_name
    sort_recom_op_name ={}
    sort_recom_op ={}
    mbs_list = args.micro_batch_size
    global_mbs_index = {}
    for i in range(len(mbs_list)):
        global_mbs_index[mbs_list[i]] = i
    # print(f"global_mbs_index: {global_mbs_index}")
    if (args.model_name == "gpt" and args.model_size == "350M") or (args.model_name == "t5" and args.model_size == "220M"):
        max_tp_size = min(args.max_tp, 4)
        max_cp_size = min(args.max_cp, 4)
    else:
        max_tp_size = min(args.max_tp, 8)
        max_cp_size = min(args.max_cp, 8)
    algo_list = [0] if args.model_name == "t5" else [0, 1]

    tp_size_list = []
    tp = 1
    while tp <= max_tp_size:
        tp_size_list.append(tp)
        tp *= 2
    
    cp_size_list = []
    cp = 1
    while cp <= max_cp_size:
        cp_size_list.append(cp)
        cp *= 2
    # print("max_tp_size: ", max_tp_size)
    # print("tp_size_list: ", tp_size_list)
    for gpu_type in args.gpu_type_set:
        sort_recom_op[gpu_type] = {}
        sort_recom_op_name[gpu_type] = {}
        for i in range(len(mbs_list)):
            sort_recom_op[gpu_type][i] = {}
            sort_recom_op_name[gpu_type][i] = {}
            for j in range(len(tp_size_list)):     
                sort_recom_op[gpu_type][i][j] = {} 
                sort_recom_op_name[gpu_type][i][j] = {}          
                for c in range(len(cp_size_list)):     
                    sort_recom_op[gpu_type][i][j][c] = {} 
                    sort_recom_op_name[gpu_type][i][j][c] = {}          
                    for k in range(len(algo_list)):
                        sort_recom_op[gpu_type][i][j][c][k] = {}
                        sort_recom_op_name[gpu_type][i][j][c][k] = {}
                        for op_name in op_list:
                            sort_recom_op[gpu_type][i][j][c][k][op_name]= activations[op_name][i][j][c][k] / compute_fwd_time[gpu_type][op_name][i][j][c][k]
    for gpu_type in args.gpu_type_set:
        for i in range(len(mbs_list)):
            for j in range(len(tp_size_list)):
                for c in range(len(cp_size_list)):
                    for k in range(len(algo_list)):
                        # 根据saved activations/comp排序名字
                        sort_recom_op_name[gpu_type][i][j][c][k] = sorted(sort_recom_op[gpu_type][i][j][c][k], key=sort_recom_op[gpu_type][i][j][c][k].get, reverse=True)                
                        
def get_recom_op_name_sort(gpu_type, mbs_index, tp_index, cp_index, algo_index):
    global sort_recom_op_name
    max_tp_index = len(sort_recom_op_name[gpu_type][mbs_index])-1
    if(max_tp_index < tp_index):
        tp_index = max_tp_index #TODO
        # sort_recom_op_name[gpu_type][mbs_index][tp_index][algo_index] = sort_recom_op_name[gpu_type][mbs_index][max_tp_index][algo_index]

    return sort_recom_op_name[gpu_type][mbs_index][tp_index][cp_index][algo_index]