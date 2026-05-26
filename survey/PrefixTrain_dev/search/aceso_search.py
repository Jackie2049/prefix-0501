# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from aceso_cost_model import read_profiled_time, predict_time_breakdown, update_recompute, get_reserved_memory_list
from multiprocessing import Process, Queue
from aceso_utils import *
from aceso_prims import action_resource_table, finetune_dim_stage_level, finetune, get_explored_cases, reset_explored_cases
from aceso_policy import *
import copy 
import time 
import os 
import csv
from aceso_prims import reset_move_count, get_move_count

args = parse_args()
print_args(args)

read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)

config_visited = {}

def initialize_search(num_stages):
    global current_min_time, unexplored_configs, explored_configs
    print(f"working on num_stages = {num_stages}")        
    config = generate_initial_config(num_stages, args)  # 根据args生成一个config
    if config is not None:
        update_recompute(config) # 根据内存大小更新config的recompute
        predict_time_breakdown(config)    # 预测config的时间
        print_simple_config_info(config, info="start", print_debug_info=args.print_debug_info, add_history=True) 
        if max(config.memory_list) < args.memory_limit:
            current_min_time = max(config.time_list)
        else:
            current_min_time = MAX_VALUE

        reset_explored_cases()
        reset_hit_resources()
        unexplored_configs = []
        explored_configs = []
    return config

def take_action(config, target_stage, prim):
    timers("prim_" + prim.name).start()
    new_config = prim.apply_to(config, target_stage)# 对config进行prim操作
    unvisited_configs = []
    if new_config is not None:
        debug_info(f">>>> {prim.name} success.", args.print_debug_info)
        if not is_visited(config_visited, hash_str=config_details(new_config, get_string=True)):
            predict_time_breakdown(new_config) # 预测新的config的时间
            unvisited_configs.append(new_config)
            print_simple_config_info(new_config, info=f"config after action {prim.name} is not visited.", print_debug_info=args.print_debug_info)
        else:
            # if(max(new_config.time_list)-5920.0<1):
            #     #save to json
            #     print("save to json")
            #     dump_config_to_json(new_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{new_config.num_stages}stages_{args.config_suffix}_test.json', args)
            print_simple_config_info(new_config, info=f"config after action {prim.name} is visited.", print_debug_info= args.print_debug_info)
    else:
        debug_info(f">>>> {prim.name} failed.", args.print_debug_info)
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
        if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
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
        if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit and config.adaptive_times < args.adaptive_hyper_parameters:
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

def multi_hop_search(config, hop_index, initial_time, adaptive_flag):
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
            action_succeed_configs = take_action(config, bottleneck, action)
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
            if _config_time < min_time and _config_memory <= args.memory_limit: # 如果新的config的时间小于最小时间并且内存小于内存限制 则更新最小时间和最小时间的config
                min_time = _config_time
                min_time_config = _config 

        if min_time_config is not None: # 当有一个action能够找到一个更优解时
            return min_time_config, hop_index + 1
        else: # 当没有一个action能够找到一个更优解时 则继续搜索
            _new_configs_all = sort_configs(new_configs_all, args.sort_metric)# 使用插入排序对新的config按照max_stage_time进行排序
            for _config in _new_configs_all:
                config_list.append(_config)        
                next_config, next_config_hop_index = multi_hop_search(_config, hop_index + 1, initial_time, adaptive_flag=False)
                if next_config is not None:
                    if max(next_config.time_list) < initial_time and max(next_config.memory_list) <= args.memory_limit:
                        return next_config, next_config_hop_index
                    else:
                        config_list.append(next_config)
    
    if args.continue_when_fail:
        min_time = MAX_VALUE
        best_config = None
        for config in config_list:
            if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
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

def trial(config, num_trial, adaptive_flag, initial_time=None):
    global timers
    timers("trial-time").start()
    reset_visited_partners()

    if initial_time is None:
        if max(config.memory_list) > args.memory_limit: # 如果config的内存大于内存限制,即没有合适的config
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
        new_config, hop_index = multi_hop_search(config, 0, initial_time, adaptive_flag) # 多跳搜索 hop_index表示搜索到该config的深度

        if new_config is not None:
            config_list.append(new_config)

            new_time = max(new_config.time_list)
            new_memory = max(new_config.memory_list)

            dec_time_gap = initial_time - new_time
            if dec_time_gap > 0 and new_memory <= args.memory_limit: # 如果新的时间小于初始时间并且内存小于内存限制 则返回新的config
                timers("trial-time").reset() 
                return new_config, num_targets, hop_index

    timers("trial-time").reset()

    min_time = MAX_VALUE
    best_config = None
    for config in config_list:
        if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit: # 如果config的时间小于最小时间并且内存小于内存限制 则返回config
            min_time = max(config.time_list)
            best_config = config 
    if best_config is None: #如果没有找到合适的config 则返回内存最小的config

        min_memory = MAX_VALUE
        for config in config_list:
            if max(config.memory_list) < min_memory:
                min_memory = max(config.memory_list)
                best_config = config 

    return best_config, max_num_targets, args.max_num_hops + 1

def run_search(num_stages, queue=None):
    global current_min_time

    config = initialize_search(num_stages) # 根据num_stages初始化一个合理的config
    if config is None:
        debug_info(f"No feasible solution for # stage {num_stages}", args.print_debug_info)
        return None

    timers("total-time").start()

    current_memory = max(config.memory_list)
    best_config = None 
    num_trial = 0
    search_time_list = [0]
    config_time_list = [current_min_time]
    num_targets_list = []
    num_hops_list = []

    if max(config.memory_list) < args.memory_limit:
        best_config = config

    adaptive_flag = False
    # 当num_trial小于最大尝试次数并且搜索时间小于总时间预算时
    while num_trial < args.max_num_trials and sum(search_time_list) < args.time_budget_total: 
        trial_start_time = time.time()
        if(args.print_debug_info):
            print_simple_config_info(config, info=f"\n[ Trial {num_trial} ] ", print_debug_info=args.print_debug_info, add_history=True)
        # 进行一次trial
        new_config, num_targets, num_hops = trial(config, num_trial, adaptive_flag, current_min_time)
        trial_end_time = time.time()
        search_time_list.append(trial_end_time - trial_start_time)
        if args.finetune_after_trial > 0:
            for _ in range(args.finetune_after_trial):
                # 对新的config进行finetune
                new_config = finetune(new_config)
                if(args.print_debug_info):
                    print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after finetune):\n", print_debug_info=args.print_debug_info, add_history=True)        
        if args.finetune_tp_dp_after_trial:
            new_config = finetune_dim_stage_level(new_config)   
            if(args.print_debug_info):
                print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after tune tp/dp) :\n", print_debug_info=args.print_debug_info, add_history=True)         

        new_time = max(new_config.time_list)
        new_memory = max(new_config.memory_list)
        config_time_list.append(new_time)

        adaptive_flag = False # 适应性标志 用于判断是否进入适应性模式 只有在没有找到更优解时才会进入适应性模式
        # 如果新的config的时间小于当前最小时间并且内存小于内存限制 或者 当前内存大于内存限制并且新的内存小于当前内存
        if (new_time < current_min_time and new_memory <= args.memory_limit) or \
            (current_memory > args.memory_limit and new_memory < current_memory):
            current_memory = new_memory
            best_config = copy.deepcopy(new_config)
            config = new_config
            if num_hops <= args.max_num_hops:
                num_targets_list.append(num_targets)
                num_hops_list.append(num_hops)    
            if new_memory <= args.memory_limit:
                current_min_time = new_time
        else:
            if args.continue_when_fail:
                # print("unexplored_configs", len(unexplored_configs))
                config = get_candidate_config(unexplored_configs)
                if config is None:
                    debug_info(f"[trail {num_trial}] config is None. enter adaptive mode.", args.print_debug_info)
                    if args.adaptive_hyper_parameters > 0:
                        adaptive_flag = True # 进入适应性模式 因为没有找到更优解
                        config = get_adaptive_config(explored_configs)
                        if config is None:
                            break
                    else:
                        break
            else:
                break

        num_trial += 1
        debug_info(f"[current best time] = {current_min_time}, num_explored_cases = {get_explored_cases()}", args.print_debug_info)

    if best_config is not None:
        dump_config_to_json(best_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{best_config.num_stages}stages_{args.config_suffix}.json', args)
    
    print_search_details(best_config, args, num_stages, num_targets_list, num_hops_list, search_time_list, config_time_list, get_reserved_memory_list(best_config), get_explored_cases())
    timers("total-time").reset()

    config_mem = 0
    if best_config is not None:
        config_mem = max(best_config.memory_list)

    if queue is not None:
        result_dict = queue.get()
        result_dict[num_stages] = current_min_time, config_mem, get_explored_cases(), sum(search_time_list), get_hit_resources()
        queue.put(result_dict)
        return 
    else:
        return current_min_time, config_mem, get_explored_cases(), sum(search_time_list), get_hit_resources()

if __name__=='__main__':

    overall_start = time.time()
    result_dict = {}
    if args.multi_process:
        process_list = []
        queue = Queue()
        queue.put(result_dict)
        for num_stages in range(args.start_num_stages, args.end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
            # if(num_stages!=4):
            #     continue
            p = Process(target=run_search, args=(num_stages, queue))
            process_list.append(p)
            p.start()
        for p in process_list:
            p.join()
        result_dict = queue.get()
    else:
        for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
            result_dict[num_stages] = run_search(num_stages, queue=None)
    overall_end = time.time()
    save_and_print_top_configs(result_dict, args)

    file_name = f'{args.log_path}{args.model_name}_{args.model_size}_summary_{args.config_suffix}.csv'
    info_to_csv = [["search_cost"], [f"{(overall_end - overall_start):.2f}"]]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)

    print(f"\noverall search time: {(overall_end - overall_start):.2f} s")
