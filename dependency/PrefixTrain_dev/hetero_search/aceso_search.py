# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from aceso_cost_model import read_profiled_time, predict_time_breakdown, update_recompute, get_reserved_memory_list,get_sort_op_by_size_comp,read_profiled_time_,set_profiled_time
from multiprocessing import Process, Queue
import multiprocessing
from aceso_utils import *
from aceso_prims import action_resource_table, finetune_dim_stage_level, finetune, get_explored_cases, reset_explored_cases,get_mean_op_success_bisection_count,get_mean_op_success_linear_count
from aceso_policy import *
import copy 
import time 
import os 
import csv
from aceso_prims import reset_move_count, get_move_count
from metis_utils import get_device_group_list ,find_combinations
args = parse_args()

# read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)
# get_sort_op_by_size_comp(args)
config_visited = {}
MIN_TIME_GET_BEST_CONFIG = 0
def initialize_search(num_stages,node_order_idx,args):
    global current_min_time, unexplored_configs, explored_configs
    print(f"working on num_stages = {num_stages}")        
    config = generate_initial_config(num_stages,node_order_idx, args)  # 根据args生成一个config
    if config is not None:
        update_recompute(config) # 根据内存大小更新config的recompute
        predict_time_breakdown(config,args=args,print_memory=True)    # 预测config的时间
        print_simple_config_info(config, info="start", print_debug_info=args.print_debug_info, add_history=True) 
        # if max(config.memory_list) < args.memory_limit:
        if check_memory_legacy(args,config):
            current_min_time = max(config.time_list)
        else:
            current_min_time = MAX_VALUE

        reset_explored_cases()
        reset_hit_resources()
        unexplored_configs = []
        explored_configs = []
    return config

def initialize_search_mamual(num_stages,node_order_idx,device_group,args):
    global current_min_time, unexplored_configs, explored_configs
    print(f"working on num_stages = {num_stages}")        
    config = generate_initial_config_v1(num_stages,node_order_idx, device_group=device_group, args=args) 
    if config is not None:
        update_recompute(config) # 根据内存大小更新config的recompute
        predict_time_breakdown(config,args=args,print_memory=False)    # 预测config的时间
        print_simple_config_info(config, info="start", print_debug_info=args.print_debug_info, add_history=True) 
        # if max(config.memory_list) < args.memory_limit:
        if check_memory_legacy(args,config):
            current_min_time = max(config.time_list)
        else:
            current_min_time = MAX_VALUE

        reset_explored_cases()
        reset_hit_resources()
        unexplored_configs = []
        explored_configs = []
    return config


def take_action(config, target_stage, prim,args=None):
    timers("prim_" + prim.name).start()
    new_config = prim.apply_to(config, target_stage,args)# 对config进行prim操作
    unvisited_configs = []
    if new_config is not None:
        # debug_info(f">>>> {prim.name} success.", args.print_debug_info)
        if not is_visited(config_visited, hash_str=config_details(new_config, get_string=True)):
            timers("prim_predict_time_breakdown2").start()
            predict_time_breakdown(new_config,args=args) # 预测新的config的时间
            timers("prim_predict_time_breakdown2").stop()
            unvisited_configs.append(new_config)
            # print_simple_config_info(new_config, info=f"config after action {prim.name} is not visited.", print_debug_info=args.print_debug_info)
        else:
            # if(max(new_config.time_list)-5920.0<1):
            #     #save to json
            #     print("save to json")
            #     dump_config_to_json(new_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{new_config.num_stages}stages_{args.config_suffix}_test.json', args)
            print_simple_config_info(new_config, info=f"config after action {prim.name} is visited.", print_debug_info= args.print_debug_info)
    # else:
    #     debug_info(f">>>> {prim.name} failed.", args.print_debug_info)
    timers("prim_" + prim.name).stop()
    return unvisited_configs

unexplored_configs = []
explored_configs = []
def get_candidate_config(candidate_set):
    if len(candidate_set) == 0:
        return None

    min_time = MAX_VALUE
    best_config = None
    for config in candidate_set:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
        if max(config.time_list) < min_time and (check_memory_legacy(args,config)==True):
            min_time = max(config.time_list)
            best_config = config 
    if best_config is None:
        min_memory = MAX_VALUE
        for config in candidate_set:
            if max(config.memory_list) < min_memory:
                min_memory = max(config.memory_list)
                best_config = config         
    candidate_set.remove(best_config)
    return best_config

def get_adaptive_config(candidate_set):
    if len(candidate_set) == 0:
        return None

    min_time = MAX_VALUE
    best_config = None
    for config in candidate_set:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit and config.adaptive_times < args.adaptive_hyper_parameters:
        if max(config.time_list) < min_time and (check_memory_legacy(args,config)==True) and config.adaptive_times < args.adaptive_hyper_parameters:
            min_time = max(config.time_list)
            best_config = config 
    if best_config is None:
        min_memory = MAX_VALUE
        for config in candidate_set:
            if max(config.memory_list) < min_memory and config.adaptive_times < args.adaptive_hyper_parameters:
                min_memory = max(config.memory_list)
                best_config = config         
    if best_config is not None:
        best_config.adaptive_times += 1
    return best_config

def multi_hop_search(config, hop_index, initial_time, adaptive_flag,args=None):
    global current_min_time
    ## reach hop limit
    if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial or \
        timers("total-time").elapsed_since_first_invoke() >= args.time_budget_total or \
        hop_index == args.max_num_hops:
        return None, None
    mark_visited(config_visited, hash_str=config_details(config, get_string=True))
    if not adaptive_flag:
        if args.adaptive_hyper_parameters > 0:
            explored_configs.append(config)
        if args.continue_when_fail and config in unexplored_configs:
            unexplored_configs.remove(config)
    ## get bottleneck 获得当前config的bottleneck的stage
    bottleneck = get_target_stage(config, adaptive_flag=adaptive_flag)
    if bottleneck is None:
        return None, None
    # 根据bottleneck获得可以解决bottleneck的actions

    '''
    action_resource_table = [
    AcesoPrim(name = "dec_op",  time = "-", memory = "-", num_devices = "0", workloads = "-", efficiency = "0", comm = "0", func = prim_mig_op if not args.simple_prim_mig else prim_mig_op_simple),
    AcesoPrim(name = "inc_dp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    AcesoPrim(name = "dec_dp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    AcesoPrim(name = "inc_tp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    AcesoPrim(name = "dec_tp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    AcesoPrim(name = "inc_mbs", time = "-", memory = "+", num_devices = "0", workloads = "0", efficiency = "+", comm = "0", func = prim_mbs),
    AcesoPrim(name = "dec_mbs", time = "+", memory = "-", num_devices = "0", workloads = "0", efficiency = "-", comm = "0", func = prim_mbs)
]'''
    actions_all, turn_back_actions, stage_type = get_actions(config, bottleneck, action_resource_table, adaptive_flag=adaptive_flag)
    print_simple_config_info(config, info=f">>", print_debug_info=args.print_debug_info)
    debug_info(f">> hop_index {hop_index}, target_stage ({bottleneck}) [{stage_type}] actions: {[[prim.name for prim in actions_] for actions_ in actions_all]}. turn back actions: {turn_back_actions}", args.print_debug_info)

    config_list  = [config]
    for actions in actions_all:
        new_configs_all = []
        for action in actions:
            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial or timers("total-time").elapsed_since_first_invoke() >= args.time_budget_total:
                break
            action_succeed_configs = take_action(config, bottleneck, action,args)
            # print("len(action_succeed_configs)", len(action_succeed_configs))
            for succeed_config in action_succeed_configs: # only one config or no config
                new_configs_all.append(succeed_config)   
                if(args.print_debug_info):
                    print_simple_config_info(succeed_config, info=f">>>> [CONFIG: target({bottleneck}), prim({action.name})]\n", print_debug_info=args.print_debug_info, add_history=True)

        min_time = initial_time
        min_time_config = None
        for _config in new_configs_all:
            _config_time = max(_config.time_list)
            _config_memory = max(_config.memory_list)
            if args.continue_when_fail and _config_time < current_min_time * 1.2: # 如果新的config的时间小于当前最小时间的1.2倍 则加入unexplored_configs
                unexplored_configs.append(_config)               
            # if _config_time < min_time and _config_memory <= args.memory_limit: # 如果新的config的时间小于最小时间并且内存小于内存限制 则更新最小时间和最小时间的config
            if _config_time < min_time and check_memory_legacy(args,_config)==True:
                global MIN_TIME_GET_BEST_CONFIG
                # print("config_time",_config_time, "MIN_TIME_GET_BEST_CONFIG",MIN_TIME_GET_BEST_CONFIG)
                MIN_TIME_GET_BEST_CONFIG = timers("total-time").elapsed_since_first_invoke()
                min_time = _config_time
                min_time_config = _config 

        if min_time_config is not None: # 当有一个action能够找到一个更优解时
            return min_time_config, hop_index + 1
        else: # 当没有一个action能够找到一个更优解时 则继续搜索
            _new_configs_all = sort_configs(new_configs_all, args.sort_metric)# 使用插入排序对新的config按照max_stage_time进行排序
            for _config in _new_configs_all:
                config_list.append(_config)        
                next_config, next_config_hop_index = multi_hop_search(_config, hop_index + 1, initial_time, adaptive_flag=False,args=args) # 递归搜索
                if next_config is not None:
                    # if max(next_config.time_list) < initial_time and max(next_config.memory_list) <= args.memory_limit:
                    if max(next_config.time_list) < initial_time and check_memory_legacy(args,next_config)==True:
                        return next_config, next_config_hop_index
                    else:
                        config_list.append(next_config)
    
    if args.continue_when_fail:
        min_time = MAX_VALUE
        best_config = None
        for config in config_list:
            # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
            if max(config.time_list) < min_time and check_memory_legacy(args,config)==True:
                min_time = max(config.time_list)
                best_config = config 
        if best_config is None:
            min_memory = MAX_VALUE
            for config in config_list:
                if max(config.memory_list) < min_memory:
                    min_memory = max(config.memory_list)
                    best_config = config 
        return best_config, args.max_num_hops
    else:
        return None, None

def trial(config, num_trial, adaptive_flag, initial_time=None,args=None):
    global timers
    timers("trial-time").start()
    reset_visited_partners()

    if initial_time is None:
        # if max(config.memory_list) > args.memory_limit: # 如果config的内存大于内存限制,即没有合适的config
        if check_memory_legacy(args,config)==False:
            initial_time = MAX_VALUE # 初始时间为最大值
        else:
            initial_time = max(config.time_list) # 否则初始时间为config的时间

    if args.only_top_1_target:
        max_num_targets = 1
    else:
        max_num_targets = config.num_stages

    num_targets = 0
    config_list = [config]
    while num_targets < max_num_targets:
        print_simple_config_info(config, info=f">>[target# {num_targets}/{max_num_targets}]", print_debug_info=args.print_debug_info)
        num_targets += 1
        new_config, hop_index = multi_hop_search(config, 0, initial_time, adaptive_flag,args=args) # 多跳搜索 hop_index表示搜索到该config的深度

        if new_config is not None:
            config_list.append(new_config)

            new_time = max(new_config.time_list)
            new_memory = max(new_config.memory_list)

            dec_time_gap = initial_time - new_time
            # if dec_time_gap > 0 and new_memory <= args.memory_limit: # 如果新的时间小于初始时间并且内存小于内存限制 则返回新的config
            if dec_time_gap > 0 and check_memory_legacy(args,new_config)==True:
            
                timers("trial-time").reset() 
                return new_config, num_targets, hop_index

    timers("trial-time").reset()

    min_time = MAX_VALUE
    best_config = None
    for config in config_list:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit: # 如果config的时间小于最小时间并且内存小于内存限制 则返回config
        if max(config.time_list) < min_time and check_memory_legacy(args,config)==True:
            min_time = max(config.time_list)
            best_config = config 
    if best_config is None: #如果没有找到合适的config 则返回内存最小的config

        min_memory = MAX_VALUE
        for config in config_list:
            if max(config.memory_list) < min_memory:
                min_memory = max(config.memory_list)
                best_config = config 

    return best_config, max_num_targets, args.max_num_hops + 1

global profile_time_global
profile_time_global = None

def run_search(num_stages, node_order_idx, device_group_idx, profile_time ,shared_dict=None,args_=None):
    # global args
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band 
    (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band )= profile_time
    set_profiled_time(profile_time)
    global current_min_time
    global args 
    global global_args
    # print(f"args_.gpu_type_num_dict" ,args_.gpu_type_num_dict)
    args= args_
    # print(f"args.gpu_type_num_dict" ,args.gpu_type_num_dict)
    global_args = args
    get_sort_op_by_size_comp(args)

    global profile_time_global
    profile_time_global = profile_time
    args.node_order = args.node_order_list[node_order_idx]
    device_group = args.total_device_group_list[num_stages][device_group_idx]
    args.num_gpus_per_node_list=[]
    args.gpu_type_list = []
    for key in args.node_order.keys():
        args.num_gpus_per_node_list.append(args.node_order[key]["GPU_NUM"])
        args.gpu_type_list.append(args.node_order[key]["GPU"])
    config = initialize_search_mamual(num_stages,node_order_idx,device_group,args)
    if config is None:
        debug_info(f"No feasible solution for # stage {num_stages}", args.print_debug_info)
        return None
    
    timers("total-time").start()

    current_memory = max(config.memory_list)
    temp_config = config
    best_config = None 
    num_trial = 0
    search_time_list = [0]
    config_time_list = [current_min_time]
    num_targets_list = []
    num_hops_list = []

    # if max(config.memory_list) < args.memory_limit:
    if check_memory_legacy(args,config):
        best_config = config

    adaptive_flag = False
    # 当num_trial小于最大尝试次数并且搜索时间小于总时间预算时
    while num_trial < args.max_num_trials and sum(search_time_list) < args.time_budget_total: 
        trial_start_time = time.time()
        if(args.print_debug_info):
            print_simple_config_info(config, info=f"\n[ Trial {num_trial} ] ", print_debug_info=args.print_debug_info, add_history=True)
        # 进行一次trial
        new_config, num_targets, num_hops = trial(config, num_trial, adaptive_flag, current_min_time,args=args)
        memory_list = new_config.memory_list
        each_stage_memory_breakdown = new_config.each_stage_memory_breakdown
        assert memory_list[0] == each_stage_memory_breakdown[0][0], f"memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        trial_end_time = time.time()
        search_time_list.append(trial_end_time - trial_start_time)
        if args.finetune_after_trial > 0:
            for _ in range(args.finetune_after_trial):
                # 对新的config进行finetune
                new_config = finetune(new_config,args=args)
                if(args.print_debug_info):
                    print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after finetune):\n", print_debug_info=args.print_debug_info, add_history=True)         
        if args.finetune_tp_dp_after_trial:
            new_config = finetune_dim_stage_level(new_config)   
            if(args.print_debug_info):
                print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after tune tp/dp) :\n", print_debug_info=args.print_debug_info, add_history=True)         
        memory_list = new_config.memory_list
        each_stage_memory_breakdown = new_config.each_stage_memory_breakdown
        assert memory_list[0] == each_stage_memory_breakdown[0][0], f"1memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        new_time = max(new_config.time_list)
        new_memory = max(new_config.memory_list)
        config_time_list.append(new_time)

        adaptive_flag = False # 适应性标志 用于判断是否进入适应性模式 只有在没有找到更优解时才会进入适应性模式
        # 如果新的config的时间小于当前最小时间并且内存小于内存限制 或者 当前内存大于内存限制并且新的内存小于当前内存
        # if (new_time < current_min_time and new_memory <= args.memory_limit) or \
        #     (current_memory > args.memory_limit and new_memory < current_memory):


        if (new_time < current_min_time and check_memory_legacy(args,new_config)==True) or \
            (check_memory_legacy(args,temp_config)==False and new_memory < current_memory):

            if(new_time < current_min_time and check_memory_legacy(args,new_config)==True):
                best_config = copy.deepcopy(new_config)
            current_memory = new_memory
            config = new_config
            if num_hops <= args.max_num_hops:
                num_targets_list.append(num_targets)
                num_hops_list.append(num_hops)    
            # if new_memory <= args.memory_limit:
            if check_memory_legacy(args,new_config):
                current_min_time = min(new_time, current_min_time)
        else:
            if args.continue_when_fail:
                # print("unexplored_configs", len(unexplored_configs))
                timers("candidate-time").start()
                config = get_candidate_config(unexplored_configs)
                timers("candidate-time").stop()
                if config is None:
                    debug_info(f"[trail {num_trial}] config is None. enter adaptive mode.", args.print_debug_info)
                    if args.adaptive_hyper_parameters > 0:
                        adaptive_flag = True # 进入适应性模式 因为没有找到更优解
                        timers("fail-time").start()
                        config = get_adaptive_config(explored_configs)
                        timers("fail-time").stop()
                        if config is None:
                            timers("fail-fail-time").start()
                            timers("fail-fail-time").stop()

                            break
                    else:
                        break
            else:
                break
        num_trial += 1
        debug_info(f"[current best time] = {current_min_time}, num_explored_cases = {get_explored_cases()}", args.print_debug_info)
    if best_config is not None:
        best_config.node_info = args.node_order_list[node_order_idx]
        print("best_config.time_list",max(best_config.time_list))
        # memory_list = best_config.memory_list
        # each_stage_memory_breakdown = best_config.each_stage_memory_breakdown
        # assert memory_list[0] == each_stage_memory_breakdown[0][0], f"2memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        dump_config_to_json(best_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{best_config.num_stages}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json', args)
    print_search_details(best_config, args, num_stages, node_order_idx,device_group_idx,num_targets_list, num_hops_list, search_time_list, config_time_list, get_reserved_memory_list(best_config), get_explored_cases())
    timers("total-time").reset()

    print(f"The search time for node_order_idx {node_order_idx}   num_stages {num_stages} device_group_idx {device_group_idx}  is {timers('total-time').elapsed_since_first_invoke()} s")
    # timers.log()

    config_mem = 0
    if best_config is not None:
        config_mem = max(best_config.memory_list)
    else:
        return None #fix 多线程卡死问题
    print("OP_SUCCESS_LINEAR_COUNT",get_mean_op_success_linear_count())
    print("OP_SUCCESS_BISECTION_COUNT",get_mean_op_success_bisection_count())
    

    if shared_dict is not None:
        assert current_min_time == max(best_config.time_list), f"current_min_time != max(best_config.time_list) {current_min_time} != {max(best_config.time_list)}"
        print("best_config.each_stage_memory_breakdown [memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]: \n", best_config.each_stage_memory_breakdown)
        print("best_config.each_stage_time_breakdown [eff_loss_time_per_gpu, recomp_time_per_gpu, comm_time_per_gpu, comm_time_per_gpu, comm_time_per_gpu]: \n", best_config.each_stage_time_breakdown)
        shared_dict[f"{node_order_idx}_{num_stages}_{device_group_idx}"] = (
            max(best_config.time_list), 
            config_mem, 
            get_explored_cases(), 
            sum(search_time_list), 
            get_hit_resources(),
            best_config.adaptive_times,
            best_config.time_list, 
            best_config.memory_list, 
            best_config.breakdown_eff_loss_time_per_gpu,
            best_config.breakdown_recomp_time_per_gpu,
            best_config.each_stage_time_breakdown,
            best_config.each_stage_memory_breakdown,
            MIN_TIME_GET_BEST_CONFIG,
            best_config
        )
        # result_dict[num_stages] =args.node_order, current_min_time, config_mem, get_explored_cases(), sum(search_time_list), get_hit_resources(),best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu,best_config.each_stage_time_breakdown,best_config.each_stage_memory_breakdown

        print(f"add result to queue {node_order_idx} {num_stages} {device_group_idx}")
        return None
    else:
        return current_min_time, config_mem, get_explored_cases(), sum(search_time_list), get_hit_resources(),best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu,best_config.each_stage_time_breakdown,best_config.each_stage_memory_breakdown,MIN_TIME_GET_BEST_CONFIG

if __name__=='__main__':
    overall_start = time.time()
    result_dict = {}
    print_args(args)

    profile_time = read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)


    if(args.enable_diff_order):

        args.node_order_list = generate_permutations(args.gpu_type_num_dict)
        for i in range(len(args.node_order_list)):
            result_dict[i] = {}
            for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
                result_dict[i][num_stages] = {}
        print("args.node_order_list",args.node_order_list)
    else:
        result_dict[0] = {}
        for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
            result_dict[0][num_stages] = {}
        args.node_order_list = [args.gpu_type_num_dict]
    p_idx = 0
    print("start find_combinations")
    total_search_status =0 
    total_device_group_list = {}
    for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
        device_group_list = find_combinations(num_stages,args.num_gpus,args.min_gpus_per_stage ,args.max_gpus_per_stage)
        total_device_group_list[num_stages] = device_group_list
        total_search_status += len(device_group_list)*len(args.node_order_list)
    print("total_search_status",total_search_status)
    print("total_device_group_list",total_device_group_list)
    args.total_device_group_list = total_device_group_list # num_stages -> device_group_list
    if args.multi_process:
        multiprocessing.set_start_method('spawn', force=True) ##
        process_list = []
        # queue = Queue()
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()
        total_process = 0
        process_have_done = 0
        for i in range(len(args.node_order_list)):
            for num_stages in range(args.start_num_stages, args.end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
                device_group_list = args.total_device_group_list[num_stages]
                for device_group_idx in range(len(device_group_list)):
                    p = Process(target=run_search, args=(num_stages,i,device_group_idx,profile_time,shared_dict,args))
                    p.start()
                    process_list.append(p)
                    total_process += 1
                    if((total_process-process_have_done)>= args.num_multi_process):
                        process_list[process_have_done].join()
                        process_have_done += 1
        for i in range(process_have_done,total_process):
            process_list[i].join()
        result_dict = shared_dict

    else:
        for i in range(len(args.node_order_list)):
            for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
                device_group_list = args.total_device_group_list[num_stages]
                print("node_order" , args.node_order_list[i] , "num_stages", num_stages , "device_group_list",device_group_list)
                args.device_group_list = device_group_list
                for device_group_idx in range(len(device_group_list)):
                    if(device_group_idx > args.start_dp_idx or device_group_idx < args.end_dp_idx):
                    # 将 device_group 转为 string
                        result_dict[i][num_stages] = run_search(num_stages,i, device_group_idx,None,args)
    # exit()
    overall_end = time.time()
    save_and_print_top_configs(result_dict, args)

    file_name = f'{args.log_path}{args.model_name}_{args.model_size}_summary_{args.config_suffix}.csv'
    info_to_csv = [["search_cost"], [f"{(overall_end - overall_start):.2f}"]]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)
    print(f"\noverall search time: {(overall_end - overall_start):.2f} s")
