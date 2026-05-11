# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy 
# from aceso_utils import parse_args, timers, debug_info, config_details, check_legality, MAX_VALUE, MIN_VALUE
from aceso_utils import *  #TODO can be more specific

from aceso_policy import get_target_stage, get_partner_stage
from aceso_cost_model import wrap_predict_delta_time, predict_stage_time_helper, predict_value_after_move, update_recompute, predict_time_breakdown, check_recompute, predict_stage_time, predict_stage_memory
from model_ops_info import get_tunable_op_list
import math
NUM_EXPLORED_CONFIGS = 0
import aceso_var 

# args = parse_args()
ops_tunable = get_tunable_op_list(aceso_var.args)

class AcesoPrim:
    def __init__(self, name, time, memory, num_devices, workloads, efficiency, comm, func):
        self.name = name
        self.time = time
        self.memory = memory
        self.num_devices = num_devices
        self.workloads = workloads
        self.efficiency = efficiency
        self.comm = comm
        self.func = func
    
    def apply_to(self, config, bottleneck,args):
        return self.func(config, bottleneck, action = self.name,args=args)

def get_explored_cases():
    global NUM_EXPLORED_CONFIGS
    return NUM_EXPLORED_CONFIGS

def reset_explored_cases():
    global NUM_EXPLORED_CONFIGS
    NUM_EXPLORED_CONFIGS = 0

def get_mig_goals(config, bottleneck, partner, metric):
    if metric == "time" or metric == "time_with_efficiency":
        time_list = config.compute_time_list
        if aceso_var.args.simple_prim_mig:
            attmpt_ratio = [1, 0.8, 0.6, 0.4, 0.2, 0]
        else:    
            attmpt_ratio = [0.4, 0.2, 0]
        gap = time_list[bottleneck] - time_list[partner]
        goals = [time_list[bottleneck] - gap * ratio for ratio in attmpt_ratio ]
    elif metric == "memory":
        # goals = [args.memory_limit]
        goals = [get_memory_limit(aceso_var.args,config.stages[bottleneck].node_id_num)]
    else:
        raise RuntimeError(f"metric {metric} not implemented.")

    return goals

move_set = {}
move_visit_count = 0
move_hit_count = 0

def reset_move_count():
    global move_visit_count, move_hit_count, move_set
    move_visit_count = 0
    move_hit_count = 0
    move_set = {}

def get_move_count():
    global move_visit_count, move_hit_count, move_set
    return move_visit_count, move_hit_count

def action_move_once(config, stage_bottleneck, stage_lowest, step_size=1, updated_recompute_ops=None):
    global move_visit_count, move_hit_count, move_set
    move_visit_count += 1
    move_str = config_details(config, get_string=True) + f"h{stage_bottleneck}l{stage_lowest}s{step_size}"
    if move_set.get(move_str) is not None:
        move_hit_count += 1
        _config = copy.deepcopy(move_set[move_str]) 
        _config.history = config.history
        return _config

    new_config = copy.deepcopy(config)

    if step_size == 0:
        return config

    btnk = stage_bottleneck
    btnk_right = stage_bottleneck + 1
    btnk_left = stage_bottleneck - 1

    if len(config.stages[btnk].ops) <= step_size:
        debug_info(f"STEP_SIZE ({step_size}) is too large", aceso_var.args.print_debug_info)
        return config

    if stage_bottleneck < stage_lowest: # 如果bottleneck 在 target stage 的左边
        new_config.stages[btnk_right].ops = new_config.stages[btnk].ops[-step_size:] + new_config.stages[btnk_right].ops
        new_config.stages[btnk_right].recompute_ops = [0 for _ in range(step_size)] + new_config.stages[btnk_right].recompute_ops 
        new_config.stages[btnk].ops = new_config.stages[btnk].ops[:-step_size]
        new_config.stages[btnk].recompute_ops = new_config.stages[btnk].recompute_ops[:-step_size]

        new_tp_size = [new_config.stages[btnk_right].tp_size[0] for _ in range(step_size)]
        new_cp_size = [new_config.stages[btnk_right].cp_size[0] for _ in range(step_size)]
        new_dp_size = [new_config.stages[btnk_right].dp_size[0] for _ in range(step_size)]
        new_config.stages[btnk_right].tp_size = new_tp_size + new_config.stages[btnk_right].tp_size
        new_config.stages[btnk_right].cp_size = new_cp_size + new_config.stages[btnk_right].cp_size
        new_config.stages[btnk_right].dp_size = new_dp_size + new_config.stages[btnk_right].dp_size
        new_config.stages[btnk].tp_size = new_config.stages[btnk].tp_size[:-step_size]
        new_config.stages[btnk].cp_size = new_config.stages[btnk].cp_size[:-step_size]
        new_config.stages[btnk].dp_size = new_config.stages[btnk].dp_size[:-step_size]
        new_config.stages[btnk_right].algo = new_config.stages[btnk].algo[-step_size:] + new_config.stages[btnk_right].algo
        new_config.stages[btnk].algo = new_config.stages[btnk].algo[:-step_size]

        if updated_recompute_ops is not None:
            new_config.stages[btnk].recompute_ops = updated_recompute_ops
        else:
            update_recompute(new_config, stage_bottleneck) 
        update_recompute(new_config, stage_bottleneck + 1) 

    elif stage_bottleneck > stage_lowest:
        new_config.stages[btnk_left].ops = new_config.stages[btnk_left].ops + new_config.stages[btnk].ops[:step_size]
        new_config.stages[btnk_left].recompute_ops = new_config.stages[btnk_left].recompute_ops + [0 for _ in range(step_size)]
        new_config.stages[btnk].ops = new_config.stages[btnk].ops[step_size:]
        new_config.stages[btnk].recompute_ops = new_config.stages[btnk].recompute_ops[step_size:]

        new_tp_size = [new_config.stages[btnk_left].tp_size[-1] for _ in range(step_size)]
        new_cp_size = [new_config.stages[btnk_left].cp_size[-1] for _ in range(step_size)]
        new_dp_size = [new_config.stages[btnk_left].dp_size[-1] for _ in range(step_size)]
        new_config.stages[btnk_left].tp_size = new_config.stages[btnk_left].tp_size + new_tp_size
        new_config.stages[btnk_left].cp_size = new_config.stages[btnk_left].cp_size + new_cp_size
        new_config.stages[btnk_left].dp_size = new_config.stages[btnk_left].dp_size + new_dp_size
        new_config.stages[btnk].tp_size = new_config.stages[btnk].tp_size[step_size:]
        new_config.stages[btnk].cp_size = new_config.stages[btnk].cp_size[step_size:]
        new_config.stages[btnk].dp_size = new_config.stages[btnk].dp_size[step_size:]
        new_config.stages[btnk_left].algo = new_config.stages[btnk_left].algo + new_config.stages[btnk].algo[:step_size]
        new_config.stages[btnk].algo = new_config.stages[btnk].algo[step_size:]

        if updated_recompute_ops is not None:
            new_config.stages[btnk].recompute_ops = updated_recompute_ops
        else:
            update_recompute(new_config, stage_bottleneck) 
        update_recompute(new_config, stage_bottleneck - 1) 
    else:
        return config

    move_set[move_str] = copy.deepcopy(new_config)
    return new_config 

def prim_mig_op(config, bottleneck, action = "",args=None):
    global NUM_EXPLORED_CONFIGS
    global OP_SUCCESS_LINEAR_COUNT
    # if max(config.memory_list) > args.memory_limit:
    if check_memory_legacy(args,config)==False:
        metric = "memory"
    else:
        metric = "time_with_efficiency"
    num_partners = args.num_partners_in_op_mig
    timers("prim_dec_op_get_partner_stage").start()
    partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric) for _ in range(num_partners)]
    timers("prim_dec_op_get_partner_stage").stop()
    success_flag = False
    time_is_up = False

    if args.predict_delta_time and metric == "time_with_efficiency":
        tmp_value_list = config.compute_time_list



    for mig_step in range(args.max_op_move_steps):
        step_success_flag = False
        original_bottleneck = bottleneck
        for partner in partners:
            if bottleneck is None or partner is None:
                debug_info(f"=> mig_step {mig_step}, bottleneck is None", args.print_move_op_details)
                break
            debug_info(f"=> mig_step {mig_step}, bottleneck = {bottleneck}, partner = {partner}", args.print_move_op_details)
            timers("prim_dec_op_get_mig_goals").start()
            goals = get_mig_goals(config, bottleneck, partner, metric)
            timers("prim_dec_op_get_mig_goals").stop()
            ## start migration
            for goal in goals:
                bottleneck = original_bottleneck
                debug_info(f"==> trying migrating under goal {goal} ...", args.print_move_op_details)
                _new_config = copy.deepcopy(config)
                while bottleneck != partner:
                    num_ops_list = [i for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size)]
                    found = False

                    if (args.bisection):#使用二分法 为什么时间会变长
                        left = 0
                        right = len(num_ops_list)-1
                        num_ops_result = None
                        count_bisection = 0
                        while left <= right:
                            count_bisection += 1
                            mid = (left + right) // 2
                            num_ops = num_ops_list[mid]
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                right = mid-1
                                num_ops_result = num_ops
                            elif value_after_move > goal:
                                left = mid+1
                            else:
                                num_ops_result = num_ops
                                break
                        if num_ops_result is not None:
                            OP_SUCCESS_LINEAR_COUNT+=1
                            OP_SUCCESS_BISECTION_COUNT_LIST.append(count_bisection)
                            OP_SUCCESS_LINEAR_COUNT_LIST.append(num_ops_result)
                            timers("prim_dec_op_action_move_once").start()
                            _new_config = action_move_once(_new_config, bottleneck, partner, num_ops_result, updated_recompute_ops)
                            timers("prim_dec_op_action_move_once").stop()
                            debug_info(f"======> success moving {num_ops_result} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                            found = True
                            timers("prim_dec_op_predict_stage_time_helper").start()
                            tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                            timers("prim_dec_op_predict_stage_time_helper").stop()
                    else:
                        for num_ops in num_ops_list:
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                OP_SUCCESS_LINEAR_COUNT+=1
                                OP_SUCCESS_BISECTION_COUNT_LIST.append(math.ceil(math.log(len(num_ops_list),2)))
                                OP_SUCCESS_LINEAR_COUNT_LIST.append(num_ops)
                                timers("prim_dec_op_action_move_once").start()
                                _new_config = action_move_once(_new_config, bottleneck, partner, num_ops, updated_recompute_ops)
                                timers("prim_dec_op_action_move_once").stop()
                                debug_info(f"======> success moving {num_ops} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                                found = True
                                timers("prim_dec_op_predict_stage_time_helper").start()
                                tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                                timers("prim_dec_op_predict_stage_time_helper").stop()
                                break
                    if time_is_up:
                        break
                    if found:
                        if bottleneck < partner:
                            bottleneck += 1
                        else:
                            bottleneck -= 1
                    else:
                        debug_info(f"======> fail moving any from {bottleneck}.", args.print_move_op_details)
                        break
                NUM_EXPLORED_CONFIGS += 1
                ## evaluate the new config
                if args.predict_delta_time and metric == "time_with_efficiency":
                    value_after_move = tmp_value_list[bottleneck]
                else:
                    timers("prim_dec_op_predict_value_after_move_1").start()
                    value_after_move, _ = predict_value_after_move(_new_config, bottleneck, bottleneck, 0, metric)
                    timers("prim_dec_op_predict_value_after_move_1").stop()
                if value_after_move < goal:
                    config = _new_config
                    timers("prim_dec_op_predict_time_breakdown").start()
                    predict_time_breakdown(config,args=args)
                    timers("prim_dec_op_predict_time_breakdown").stop()
                    step_success_flag = True
                    success_flag = True
                    debug_info(f"======> success at goal {goal}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                    debug_info(f"time after moving: {config.compute_time_list}", args.print_move_op_details)
                    debug_info(f"memory after moving: {config.memory_list}", args.print_move_op_details)
                    break 
                else:
                    debug_info(f"failed at bottleneck {bottleneck}, pred_value = {value_after_move}, goal = {goal}", args.print_move_op_details)
                    if time_is_up:
                        break
            if step_success_flag or time_is_up:
                break
        ## end migration
        if not step_success_flag or time_is_up:
            debug_info(f"======> failed at all goals.", args.print_move_op_details)
            break 
        timers("prim_dec_op_get_target_stage").start()
        bottleneck = get_target_stage(config, metric=metric, other_info="mig" + str(mig_step))
        partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric, other_info="mig" + str(mig_step)) for _ in range(num_partners) ]
        timers("prim_dec_op_get_target_stage").stop()
    if success_flag:
        return config
    else:
        return None

# def prim_mig_op(config, bottleneck, action = "",args=None):
#     global NUM_EXPLORED_CONFIGS
#     global OP_SUCCESS_LINEAR_COUNT
#     # if max(config.memory_list) > args.memory_limit:
#     if check_memory_legacy(args,config)==False:
#         metric = "memory"
#     else:
#         metric = "time_with_efficiency"
#     num_partners = args.num_partners_in_op_mig
#     timers("prim_dec_op_get_partner_stage").start()
#     partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric) for _ in range(num_partners)]
#     timers("prim_dec_op_get_partner_stage").stop()
#     success_flag = False
#     time_is_up = False

#     if args.predict_delta_time and metric == "time_with_efficiency":
#         tmp_value_list = config.compute_time_list



#     for mig_step in range(args.max_op_move_steps):
#         step_success_flag = False
#         original_bottleneck = bottleneck
#         for partner in partners:
#             if bottleneck is None or partner is None:
#                 debug_info(f"=> mig_step {mig_step}, bottleneck is None", args.print_move_op_details)
#                 break
#             debug_info(f"=> mig_step {mig_step}, bottleneck = {bottleneck}, partner = {partner}", args.print_move_op_details)
#             timers("prim_dec_op_get_mig_goals").start()
#             goals = get_mig_goals(config, bottleneck, partner, metric)
#             timers("prim_dec_op_get_mig_goals").stop()
#             ## start migration
#             for goal in goals:
#                 bottleneck = original_bottleneck
#                 debug_info(f"==> trying migrating under goal {goal} ...", args.print_move_op_details)
#                 _new_config = copy.deepcopy(config)
#                 while bottleneck != partner:
#                     num_ops_list = []
#                     bottleneck_op_start = 1
#                     for i in range(bottleneck):
#                         bottleneck_op_start += len(config.stages[i].ops)
#                     bottleneck_op_end = bottleneck_op_start + len(config.stages[bottleneck].ops)-1
#                     for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size):
#                         if(partner < bottleneck):
#                             if (bottleneck_op_start + i -1)%13==1 or (bottleneck_op_start + i-1)%13==9:
#                                 num_ops_list.append(i)
#                         elif(partner > bottleneck):
#                             if (bottleneck_op_end - i -1)%13==0 or (bottleneck_op_end - i -1)%13==8:
#                                 num_ops_list.append(i)
#                         else:
#                             continue
#                     found = False

#                     # for idx in range(0,len(num_ops_list),2):
#                     #     num_ops3 = num_ops_list[idx]
#                     #     if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
#                     #         time_is_up = True
#                     #         break
#                     #     if args.predict_delta_time and metric == "time_with_efficiency":
#                     #         timers("prim_dec_op_predict_delta_time").start()
#                     #         delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops3)
#                     #         updated_recompute_ops = None
#                     #         value_after_move3 = tmp_value_list[bottleneck] - delta_time
#                     #         timers("prim_dec_op_predict_delta_time").stop()
#                     #     else:
#                     #         timers("prim_dec_op_predict_value_after_move").start()
#                     #         value_after_move3, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops3, metric)
#                     #         timers("prim_dec_op_predict_value_after_move").stop()
#                     #     if value_after_move3 < goal:
#                     #         if(idx>0):
#                     #             value_after_move__ = predict_value_after_move(_new_config, bottleneck, partner, num_ops3-1, metric)[0]
#                     #             if(value_after_move__<goal):
#                     #                 num_ops3 = num_ops3-1
#                     #                 value_after_move3 = value_after_move__

#                     #         # timers("prim_dec_op_action_move_once").start()
#                     #         # _new_config = action_move_once(_new_config, bottleneck, partner, num_ops3, updated_recompute_ops)
#                     #         # timers("prim_dec_op_action_move_once").stop()
#                     #         # debug_info(f"======> success moving {num_ops3} from {bottleneck}. ({value_after_move3} < {goal} (goal))", args.print_move_op_details)
#                     #         # found = True
#                     #         # timers("prim_dec_op_predict_stage_time_helper").start()
#                     #         # tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
#                     #         # timers("prim_dec_op_predict_stage_time_helper").stop()
#                     #         break
                    
#                     left = 0
#                     right = len(num_ops_list)-1
#                     num_ops_result = None
#                     while left <= right:
#                         mid = (left + right) // 2
#                         num_ops3 = num_ops_list[mid]
#                         if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
#                             time_is_up = True
#                             break
#                         if args.predict_delta_time and metric == "time_with_efficiency":
#                             timers("prim_dec_op_predict_delta_time").start()
#                             delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops3)
#                             updated_recompute_ops = None
#                             value_after_move1 = tmp_value_list[bottleneck] - delta_time
#                             timers("prim_dec_op_predict_delta_time").stop()
#                         else:
#                             timers("prim_dec_op_predict_value_after_move").start()
#                             value_after_move1, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops3, metric)
#                             timers("prim_dec_op_predict_value_after_move").stop()
#                         if value_after_move1 < goal:
#                             right = mid-1
#                             num_ops_result = num_ops3
#                         elif value_after_move1 > goal:
#                             left = mid+1
#                         else:
#                             num_ops_result = num_ops3
#                             break
#                     # if num_ops_result is not None:
#                         # timers("prim_dec_op_action_move_once").start()
#                         # _new_config = action_move_once(_new_config, bottleneck, partner, num_ops_result, updated_recompute_ops)
#                         # timers("prim_dec_op_action_move_once").stop()
#                         # debug_info(f"======> success moving {num_ops_result} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
#                         # found = True
#                         # timers("prim_dec_op_predict_stage_time_helper").start()
#                         # tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
#                         # timers("prim_dec_op_predict_stage_time_helper").stop()

#                     for num_ops in num_ops_list:
#                         if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
#                             time_is_up = True
#                             break
#                         if args.predict_delta_time and metric == "time_with_efficiency":
#                             timers("prim_dec_op_predict_delta_time").start()
#                             delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
#                             updated_recompute_ops = None
#                             value_after_move = tmp_value_list[bottleneck] - delta_time
#                             timers("prim_dec_op_predict_delta_time").stop()
#                         else:
#                             timers("prim_dec_op_predict_value_after_move").start()
#                             value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                            
#                             timers("prim_dec_op_predict_value_after_move").stop()
#                         if value_after_move < goal:
#                             if num_ops_result is not None:

#                                 if(num_ops_result!=num_ops):
#                                     print(f"goals:{goals},metric:{metric}")
#                                     for num_ops_ in num_ops_list:
#                                         value_after_move_ = predict_value_after_move(_new_config, bottleneck, partner, num_ops_, metric,debug=True if (num_ops_==27 or num_ops_==35) else False)[0]
#                                         print(f"num_ops_:{num_ops_},value_after_move_:{value_after_move_}")

#                                     assert(num_ops_result == num_ops),f"num_ops_result:{num_ops_result},num_ops:{num_ops}:goal:{goal} ,value_after_move1 :{value_after_move1},value_after_move:{value_after_move} \
#                                     value_after_move2 { predict_value_after_move(_new_config, bottleneck, partner, num_ops_result-1, metric)[0]}\
#                                     "
#                             # assert num_ops3 == num_ops,f"num_ops3:{num_ops3},num_ops:{num_ops}:goal:{goal} ,value_after_move3 :{value_after_move3},value_after_move:{value_after_move}"
#                             timers("prim_dec_op_action_move_once").start()
#                             _new_config = action_move_once(_new_config, bottleneck, partner, num_ops, updated_recompute_ops)
#                             timers("prim_dec_op_action_move_once").stop()
#                             debug_info(f"======> success moving {num_ops} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
#                             found = True
#                             timers("prim_dec_op_predict_stage_time_helper").start()
#                             tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
#                             timers("prim_dec_op_predict_stage_time_helper").stop()
#                             break
#                     # exit()
#                     if time_is_up:
#                         break
#                     if found:
#                         if bottleneck < partner:
#                             bottleneck += 1
#                         else:
#                             bottleneck -= 1
#                     else:
#                         debug_info(f"======> fail moving any from {bottleneck}.", args.print_move_op_details)
#                         break
#                 NUM_EXPLORED_CONFIGS += 1
#                 ## evaluate the new config
#                 if args.predict_delta_time and metric == "time_with_efficiency":
#                     value_after_move = tmp_value_list[bottleneck]
#                 else:
#                     timers("prim_dec_op_predict_value_after_move_1").start()
#                     value_after_move, _ = predict_value_after_move(_new_config, bottleneck, bottleneck, 0, metric)
#                     timers("prim_dec_op_predict_value_after_move_1").stop()
#                 if value_after_move < goal:
#                     config = _new_config
#                     timers("prim_dec_op_predict_time_breakdown").start()
#                     predict_time_breakdown(config,args=args)
#                     timers("prim_dec_op_predict_time_breakdown").stop()
#                     step_success_flag = True
#                     success_flag = True
#                     debug_info(f"======> success at goal {goal}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
#                     debug_info(f"time after moving: {config.compute_time_list}", args.print_move_op_details)
#                     debug_info(f"memory after moving: {config.memory_list}", args.print_move_op_details)
#                     break 
#                 else:
#                     debug_info(f"failed at bottleneck {bottleneck}, pred_value = {value_after_move}, goal = {goal}", args.print_move_op_details)
#                     if time_is_up:
#                         break
#             if step_success_flag or time_is_up:
#                 break
#         ## end migration
#         if not step_success_flag or time_is_up:
#             debug_info(f"======> failed at all goals.", args.print_move_op_details)
#             break 
#         timers("prim_dec_op_get_target_stage").start()
#         bottleneck = get_target_stage(config, metric=metric, other_info="mig" + str(mig_step))
#         partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric, other_info="mig" + str(mig_step)) for _ in range(num_partners) ]
#         timers("prim_dec_op_get_target_stage").stop()
#     if success_flag:
#         return config
#     else:
#         return None

def prim_mig_op_block(config, bottleneck, action = "",args=None):
    global NUM_EXPLORED_CONFIGS

    # if max(config.memory_list) > args.memory_limit:
    if check_memory_legacy(args,config)==False:
        metric = "memory"
    else:
        metric = "time_with_efficiency"
    num_partners = args.num_partners_in_op_mig
    timers("prim_dec_op_get_partner_stage").start()
    partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric) for _ in range(num_partners)]
    timers("prim_dec_op_get_partner_stage").stop()
    success_flag = False
    time_is_up = False

    if args.predict_delta_time and metric == "time_with_efficiency":
        tmp_value_list = config.compute_time_list



    for mig_step in range(args.max_op_move_steps):
        step_success_flag = False
        original_bottleneck = bottleneck
        for partner in partners:
            if bottleneck is None or partner is None:
                debug_info(f"=> mig_step {mig_step}, bottleneck is None", args.print_move_op_details)
                break
            debug_info(f"=> mig_step {mig_step}, bottleneck = {bottleneck}, partner = {partner}", args.print_move_op_details)
            timers("prim_dec_op_get_mig_goals").start()
            goals = get_mig_goals(config, bottleneck, partner, metric)
            timers("prim_dec_op_get_mig_goals").stop()
            ## start migration
            for goal in goals:
                bottleneck = original_bottleneck
                debug_info(f"==> trying migrating under goal {goal} ...", args.print_move_op_details)
                _new_config = copy.deepcopy(config)
                while bottleneck != partner:
                    bottleneck_op_start = 1
                    for i in range(bottleneck):
                        bottleneck_op_start += len(config.stages[i].ops)
                    bottleneck_op_end = bottleneck_op_start + len(config.stages[bottleneck].ops)-1
                    # num_ops_list = [i for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size)]
                    num_ops_list = []
                    for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size):
                        if(partner < bottleneck):
                            if (bottleneck_op_start + i -1)%args.num_ops_each_layer==1 or (bottleneck_op_start + i-1)%args.num_ops_each_layer==9:
                                num_ops_list.append(i)
                        elif(partner > bottleneck):
                            if (bottleneck_op_end - i -1)%args.num_ops_each_layer==0 or (bottleneck_op_end - i -1)%args.num_ops_each_layer==8:
                                num_ops_list.append(i)
                        else:
                            continue

                    found = False
                    if (args.bisection):#使用二分法 为什么时间会变长
                        left = 0
                        right = len(num_ops_list)-1
                        num_ops_result = None
                        while left <= right:
                            mid = (left + right) // 2
                            num_ops = num_ops_list[mid]
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                right = mid-1
                                num_ops_result = num_ops
                            elif value_after_move > goal:
                                left = mid+1
                            else:
                                num_ops_result = num_ops
                                break
                        if num_ops_result is not None:
                            timers("prim_dec_op_action_move_once").start()
                            _new_config = action_move_once(_new_config, bottleneck, partner, num_ops_result, updated_recompute_ops)
                            timers("prim_dec_op_action_move_once").stop()
                            debug_info(f"======> success moving {num_ops_result} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                            found = True
                            timers("prim_dec_op_predict_stage_time_helper").start()
                            tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                            timers("prim_dec_op_predict_stage_time_helper").stop()
                    else:
                        for num_ops in num_ops_list:
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                # if(OP_SUCCESS_LINEAR_COUNT.get(num_ops)==None):
                                #     OP_SUCCESS_LINEAR_COUNT[num_ops]=1
                                # else:
                                #     OP_SUCCESS_LINEAR_COUNT[num_ops]+=1
                                # if(OP_SUCCESS_BISECTION_COUNT.get(math.log(len(num_ops_list),2))==None):
                                #     OP_SUCCESS_BISECTION_COUNT[math.log(len(num_ops_list),2)]=1
                                # else:
                                #     OP_SUCCESS_BISECTION_COUNT[math.log(len(num_ops_list),2)]+=1

                                timers("prim_dec_op_action_move_once").start()
                                _new_config = action_move_once(_new_config, bottleneck, partner, num_ops, updated_recompute_ops)
                                timers("prim_dec_op_action_move_once").stop()
                                debug_info(f"======> success moving {num_ops} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                                found = True
                                timers("prim_dec_op_predict_stage_time_helper").start()
                                tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                                timers("prim_dec_op_predict_stage_time_helper").stop()
                                break
                    
                    if time_is_up:
                        break
                    if found:
                        if bottleneck < partner:
                            bottleneck += 1
                        else:
                            bottleneck -= 1
                    else:
                        debug_info(f"======> fail moving any from {bottleneck}.", args.print_move_op_details)
                        break
                NUM_EXPLORED_CONFIGS += 1
                ## evaluate the new config
                if args.predict_delta_time and metric == "time_with_efficiency":
                    value_after_move = tmp_value_list[bottleneck]
                else:
                    timers("prim_dec_op_predict_value_after_move_1").start()
                    value_after_move, _ = predict_value_after_move(_new_config, bottleneck, bottleneck, 0, metric)
                    timers("prim_dec_op_predict_value_after_move_1").stop()
                if value_after_move < goal:
                    config = _new_config
                    timers("prim_dec_op_predict_time_breakdown").start()
                    predict_time_breakdown(config,args=args)
                    timers("prim_dec_op_predict_time_breakdown").stop()
                    step_success_flag = True
                    success_flag = True
                    debug_info(f"======> success at goal {goal}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                    debug_info(f"time after moving: {config.compute_time_list}", args.print_move_op_details)
                    debug_info(f"memory after moving: {config.memory_list}", args.print_move_op_details)
                    break 
                else:
                    debug_info(f"failed at bottleneck {bottleneck}, pred_value = {value_after_move}, goal = {goal}", args.print_move_op_details)
                    if time_is_up:
                        break
            if step_success_flag or time_is_up:
                break
        ## end migration
        if not step_success_flag or time_is_up:
            debug_info(f"======> failed at all goals.", args.print_move_op_details)
            break 
        timers("prim_dec_op_get_target_stage").start()
        bottleneck = get_target_stage(config, metric=metric, other_info="mig" + str(mig_step))
        partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric, other_info="mig" + str(mig_step)) for _ in range(num_partners) ]
        timers("prim_dec_op_get_target_stage").stop()
    if success_flag:
        return config
    else:
        return None


def prim_mig_op_layer(config, bottleneck, action = "",args=None):
    global NUM_EXPLORED_CONFIGS
    global OP_SUCCESS_LINEAR_COUNT
    # if max(config.memory_list) > args.memory_limit:
    if check_memory_legacy(args,config)==False:
        metric = "memory"
    else:
        metric = "time_with_efficiency"
    num_partners = args.num_partners_in_op_mig
    timers("prim_dec_op_get_partner_stage").start()
    partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric) for _ in range(num_partners)]
    timers("prim_dec_op_get_partner_stage").stop()
    success_flag = False
    time_is_up = False

    if args.predict_delta_time and metric == "time_with_efficiency":
        tmp_value_list = config.compute_time_list
    for mig_step in range(args.max_op_move_steps):
        step_success_flag = False
        original_bottleneck = bottleneck
        for partner in partners:
            if bottleneck is None or partner is None:
                debug_info(f"=> mig_step {mig_step}, bottleneck is None", args.print_move_op_details)
                break
            debug_info(f"=> mig_step {mig_step}, bottleneck = {bottleneck}, partner = {partner}", args.print_move_op_details)
            timers("prim_dec_op_get_mig_goals").start()
            goals = get_mig_goals(config, bottleneck, partner, metric)
            timers("prim_dec_op_get_mig_goals").stop()
            ## start migration
            for goal in goals:
                bottleneck = original_bottleneck
                debug_info(f"==> trying migrating under goal {goal} ...", args.print_move_op_details)
                _new_config = copy.deepcopy(config)
                while bottleneck != partner:
                    bottleneck_op_start = 1
                    for i in range(bottleneck):
                        bottleneck_op_start += len(config.stages[i].ops)
                    bottleneck_op_end = bottleneck_op_start + len(config.stages[bottleneck].ops)-1
                    # num_ops_list = [i for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size)]
                    num_ops_list = []
                    for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size):
                        if(partner < bottleneck):
                            if (bottleneck_op_start + i -1)%args.num_ops_each_layer==1:
                                num_ops_list.append(i)
                        elif(partner > bottleneck):
                            if (bottleneck_op_end - i -1)%args.num_ops_each_layer==0:
                                num_ops_list.append(i)
                        else:
                            continue
                    found = False
                    if (args.bisection):#使用二分法 为什么时间会变长
                        left = 0
                        right = len(num_ops_list)-1
                        num_ops_result = None
                        count_bisection = 0
                        while left < right:
                            count_bisection+=1
                            mid = (left + right) // 2
                            num_ops = num_ops_list[mid]
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                right = mid
                                num_ops_result = num_ops
                            elif value_after_move > goal:
                                left = mid+1
                            else:
                                num_ops_result = num_ops
                                break
                        if num_ops_result is not None:
                            OP_SUCCESS_LINEAR_COUNT+=1
                            OP_SUCCESS_BISECTION_COUNT_LIST.append(count_bisection)
                            OP_SUCCESS_LINEAR_COUNT_LIST.append(num_ops_result)
                            timers("prim_dec_op_action_move_once").start()
                            _new_config = action_move_once(_new_config, bottleneck, partner, num_ops_result, updated_recompute_ops)
                            timers("prim_dec_op_action_move_once").stop()
                            debug_info(f"======> success moving {num_ops_result} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                            found = True
                            timers("prim_dec_op_predict_stage_time_helper").start()
                            tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                            timers("prim_dec_op_predict_stage_time_helper").stop()
                    else:
                        for num_ops in num_ops_list:
                            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                                time_is_up = True
                                break
                            if args.predict_delta_time and metric == "time_with_efficiency":
                                timers("prim_dec_op_predict_delta_time").start()
                                delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                                updated_recompute_ops = None
                                value_after_move = tmp_value_list[bottleneck] - delta_time
                                timers("prim_dec_op_predict_delta_time").stop()
                            else:
                                timers("prim_dec_op_predict_value_after_move").start()
                                value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                                timers("prim_dec_op_predict_value_after_move").stop()
                            if value_after_move < goal:
                                OP_SUCCESS_LINEAR_COUNT+=1
                                OP_SUCCESS_BISECTION_COUNT_LIST.append(math.ceil(math.log(len(num_ops_list),2)))
                                OP_SUCCESS_LINEAR_COUNT_LIST.append(num_ops)
                                timers("prim_dec_op_action_move_once").start()
                                _new_config = action_move_once(_new_config, bottleneck, partner, num_ops, updated_recompute_ops)
                                timers("prim_dec_op_action_move_once").stop()
                                debug_info(f"======> success moving {num_ops} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                                found = True
                                timers("prim_dec_op_predict_stage_time_helper").start()
                                tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                                timers("prim_dec_op_predict_stage_time_helper").stop()
                                break
                    
                    if time_is_up:
                        break
                    if found:
                        if bottleneck < partner:
                            bottleneck += 1
                        else:
                            bottleneck -= 1
                    else:
                        debug_info(f"======> fail moving any from {bottleneck}.", args.print_move_op_details)
                        break
                NUM_EXPLORED_CONFIGS += 1
                ## evaluate the new config
                if args.predict_delta_time and metric == "time_with_efficiency":
                    value_after_move = tmp_value_list[bottleneck]
                else:
                    timers("prim_dec_op_predict_value_after_move_1").start()
                    value_after_move, _ = predict_value_after_move(_new_config, bottleneck, bottleneck, 0, metric)
                    timers("prim_dec_op_predict_value_after_move_1").stop()
                if value_after_move < goal:
                    config = _new_config
                    timers("prim_dec_op_predict_time_breakdown").start()
                    predict_time_breakdown(config,args=args)
                    timers("prim_dec_op_predict_time_breakdown").stop()
                    step_success_flag = True
                    success_flag = True
                    debug_info(f"======> success at goal {goal}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                    debug_info(f"time after moving: {config.compute_time_list}", args.print_move_op_details)
                    debug_info(f"memory after moving: {config.memory_list}", args.print_move_op_details)
                    break 
                else:
                    debug_info(f"failed at bottleneck {bottleneck}, pred_value = {value_after_move}, goal = {goal}", args.print_move_op_details)
                    if time_is_up:
                        break
            if step_success_flag or time_is_up:
                break
        ## end migration
        if not step_success_flag or time_is_up:
            debug_info(f"======> failed at all goals.", args.print_move_op_details)
            break 
        timers("prim_dec_op_get_target_stage").start()
        bottleneck = get_target_stage(config, metric=metric, other_info="mig" + str(mig_step))
        partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric, other_info="mig" + str(mig_step)) for _ in range(num_partners) ]
        timers("prim_dec_op_get_target_stage").stop()
    if success_flag:
        return config
    else:
        return None


## This version tries to migrate for only one step every time.
def prim_mig_op_simple(config, bottleneck, action = "",args=None):
    global NUM_EXPLORED_CONFIGS

    # if max(config.memory_list) > args.memory_limit:
    if check_memory_legacy(args,config)==False:
        metric = "memory"
    else:
        metric = "time_with_efficiency"

    num_partners = args.num_partners_in_op_mig
    partners = [get_partner_stage(config, action="mig_workloads", partner_action="", target_stage=bottleneck, metric=metric) for _ in range(num_partners)]
    success_flag = False
    time_is_up = False

    if args.predict_delta_time and metric == "time_with_efficiency":
        tmp_value_list = config.compute_time_list

    for partner in partners:
        if bottleneck is None or partner is None:
            break
        goals = get_mig_goals(config, bottleneck, partner, metric)

        for goal in goals:
            debug_info(f"==> trying migrating under goal {goal} ...", args.print_move_op_details)
            _new_config = copy.deepcopy(config)
            num_ops_list = [i for i in range(0, len(_new_config.stages[bottleneck].ops), args.op_group_size)]
            for num_ops in num_ops_list:
                if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial:
                    time_is_up = True
                    break
                if args.predict_delta_time and metric == "time_with_efficiency":
                    delta_time = wrap_predict_delta_time(_new_config, bottleneck, partner, num_ops)
                    updated_recompute_ops = None
                    value_after_move = tmp_value_list[bottleneck] - delta_time
                else:
                    value_after_move, updated_recompute_ops = predict_value_after_move(_new_config, bottleneck, partner, num_ops, metric)
                if value_after_move < goal:
                    _new_config = action_move_once(_new_config, bottleneck, partner, num_ops, updated_recompute_ops)
                    debug_info(f"======> success moving {num_ops} from {bottleneck}. ({value_after_move} < {goal} (goal))", args.print_move_op_details)
                    success_flag = True
                    tmp_value_list = [predict_stage_time_helper(_new_config, stage_index) for stage_index in range(_new_config.num_stages)]
                    break
                else:
                    debug_info(f"======> failed moving {num_ops} from {bottleneck}. ({value_after_move} > {goal} (goal))", args.print_move_op_details)
            if time_is_up:
                break
            if success_flag:
                NUM_EXPLORED_CONFIGS += 1
                config = _new_config
                predict_time_breakdown(config,args=args)
                debug_info(f"======> success at goal {goal}.", args.print_move_op_details)
                break 
        if success_flag or time_is_up:
            break
    if success_flag:
        return config
    else:
        return None



def legal_producer(config, stage_index, dim, args):
    ## GPU num check
    if config.stages[stage_index].num_gpus == 1:
        return False
    ## min tp check
    elif dim == "tp" and min(config.stages[stage_index].tp_size) == 1:
        return False 
    ## max dp check
    elif dim == "dp":
        dp_size = config.stages[stage_index].dp_size
        base_bs = config.micro_bs
        if min(dp_size) == 1:
            return False 
        for i in range(len(dp_size)):
            if base_bs // (dp_size[i] // 2) not in args.micro_batch_size:
                return False 
    return True   

def legal_consumer(config, stage_index, dim, args):
    ## GPU num check
    if config.stages[stage_index].num_gpus * 2 > args.num_gpus - (config.num_stages - 1):
        return False
    ## max tp check
    # elif dim == "tp" and max(config.stages[stage_index].tp_size) * 2 > args.max_tp:
    elif dim == "tp" and max(config.stages[stage_index].tp_size) * 2 > get_max_tp(args,config.stages[stage_index]):
        return False 
    ## max dp check
    elif dim == "dp":
        dp_size = config.stages[stage_index].dp_size
        base_bs = config.micro_bs
        for i in range(len(dp_size)):
            if base_bs // (dp_size[i] * 2) not in args.micro_batch_size:
                return False 
    return True   

def get_next_producer(config, visited_partners, max_produce_gpus=MAX_VALUE):
    producer_stage = None
    producer_dim = None

    min_time = MAX_VALUE
    for stage_index in range(config.num_stages):
        if stage_index not in visited_partners and \
            config.stages[stage_index].num_gpus // 2 <= max_produce_gpus:
            for dim in ["tp", "dp"]:
                if legal_producer(config, stage_index, dim, aceso_var.args):
                    time_after_move, _ = predict_value_after_move(config, bottleneck=stage_index, partner=None, num_ops_moved=0, metric="time", inc_gpus=False, dec_gpus=True, dim=dim)
                    if time_after_move < min_time:
                        min_time = time_after_move
                        producer_stage = stage_index
                        producer_dim = dim

    return producer_stage, producer_dim

def get_next_consumer(config, visited_partners, max_consume_gpus=MAX_VALUE):
    consumer_stage = None
    consumer_dim = None

    min_time = MAX_VALUE
    for stage_index in range(config.num_stages):
        if stage_index not in visited_partners and \
            config.stages[stage_index].num_gpus <= max_consume_gpus:
            for dim in ["tp", "dp"]:
                if legal_consumer(config, stage_index, dim, aceso_var.args):
                    time_after_move, _ = predict_value_after_move(config, bottleneck=stage_index, partner=None, num_ops_moved=0, metric="time", inc_gpus=True, dec_gpus=False, dim=dim)
                    if time_after_move < min_time:
                        min_time = time_after_move
                        consumer_stage = stage_index
                        consumer_dim = dim

    return consumer_stage, consumer_dim

def inc_op_parallelism(config, stage_index, dim):
    if dim == "tp":
        for i in range(len(config.stages[stage_index].ops)):
            config.stages[stage_index].tp_size[i] *= 2 
    elif dim == "dp":
        for i in range(len(config.stages[stage_index].ops)):
            config.stages[stage_index].dp_size[i] *= 2  
    else:
        raise RuntimeError(f"unexpected producer dim {dim}")  

def dec_op_parallelism(config, stage_index, dim):
    if dim == "tp":
        for i in range(len(config.stages[stage_index].ops)):
            config.stages[stage_index].tp_size[i] //= 2 
    elif dim == "dp":
        for i in range(len(config.stages[stage_index].ops)):
            config.stages[stage_index].dp_size[i] //= 2  
    else:
        raise RuntimeError(f"unexpected producer dim {dim}")   

def prim_tp_dp(config, bottleneck, action,args):
    global NUM_EXPLORED_CONFIGS

    if config.num_stages <= 1:
        return None    

    consumer_stage = None
    producer_stage = None
    if "inc" in action:
        consumer_stage = bottleneck
        consumer_dim = action.split("inc_")[1]
        consume_num_gpus = config.stages[bottleneck].num_gpus 
        debug_info(f"[DEBUG action_tp_dp] [initial] consumer_stage: {consumer_stage}, consumer_dim: {consumer_dim}", args.print_gpu_mig_details)
        if not legal_consumer(config, consumer_stage, consumer_dim, args):
            return None
    elif "dec" in action:
        producer_stage = bottleneck
        producer_dim = action.split("dec_")[1]
        produce_num_gpus = config.stages[bottleneck].num_gpus // 2
        debug_info(f"[DEBUG action_tp_dp] [initial] producer_stage: {producer_stage}, producer_dim: {producer_dim}", args.print_gpu_mig_details)
        if not legal_producer(config, producer_stage, producer_dim, args):
            return None

    best_config = None
    visited_stages = []
    for tried_times in range(config.num_stages - 1):
        NUM_EXPLORED_CONFIGS += 1   
        partner_stages = []
        partner_actions = []

        if "inc" in action:
            producer_stage, producer_dim = get_next_producer(config, visited_stages + [bottleneck])
            debug_info(f"[DEBUG action_tp_dp] producer_stage: {producer_stage}, producer_dim: {producer_dim}", args.print_gpu_mig_details)
            visited_stages.append(producer_stage)
            if producer_stage is None:
                break
            _produce_num_gpus = config.stages[producer_stage].num_gpus // 2
            _consume_num_gpus = consume_num_gpus
        elif "dec" in action:
            consumer_stage, consumer_dim = get_next_consumer(config, visited_stages + [bottleneck])
            debug_info(f"[DEBUG action_tp_dp] consumer_stage: {consumer_stage}, consumer_dim: {consumer_dim}", args.print_gpu_mig_details)
            visited_stages.append(consumer_stage)
            if consumer_stage is None:
                break
            _consume_num_gpus = config.stages[consumer_stage].num_gpus
            _produce_num_gpus = produce_num_gpus

        if _produce_num_gpus >= _consume_num_gpus:
            new_config = copy.deepcopy(config)
            new_config.stages[producer_stage].num_gpus -= _produce_num_gpus
            dec_op_parallelism(new_config, producer_stage, producer_dim)          
            if producer_stage != bottleneck:
                partner_stages.append(producer_stage)
                partner_actions.append(f"dec_{producer_dim}")                

            current_consume_num_gpus = _consume_num_gpus
            current_consumer_stage = consumer_stage
            current_consumer_dim = consumer_dim

            while _produce_num_gpus > 0:
                new_config.stages[current_consumer_stage].num_gpus += current_consume_num_gpus # x2
                inc_op_parallelism(new_config, current_consumer_stage, current_consumer_dim) # tp_size/dp_size x2
                _produce_num_gpus -= current_consume_num_gpus # reduce the _produce_num_gpus
                if current_consumer_stage != bottleneck:
                    partner_stages.append(current_consumer_stage)
                    partner_actions.append(f"inc_{current_consumer_dim}")
                if _produce_num_gpus > 0:
                    current_consumer_stage, current_consumer_dim = get_next_consumer(config, partner_stages + [bottleneck], _produce_num_gpus)
                    debug_info(f"[DEBUG action_tp_dp] consumer_stage: {current_consumer_stage}, consumer_dim: {current_consumer_dim}", args.print_gpu_mig_details)
                    if current_consumer_stage is None:
                        break
                    current_consume_num_gpus = config.stages[current_consumer_stage].num_gpus

        elif _produce_num_gpus < _consume_num_gpus:
            new_config = copy.deepcopy(config)
            new_config.stages[consumer_stage].num_gpus += _consume_num_gpus
            inc_op_parallelism(new_config, consumer_stage, consumer_dim)
            if consumer_stage != bottleneck:
                partner_stages.append(consumer_stage)
                partner_actions.append(f"inc_{consumer_dim}")     

            current_producer_stage = producer_stage
            current_produce_num_gpus = _produce_num_gpus
            current_producer_dim = producer_dim

            while _consume_num_gpus > 0:
                new_config.stages[current_producer_stage].num_gpus -= current_produce_num_gpus
                dec_op_parallelism(new_config, current_producer_stage, current_producer_dim)
                _consume_num_gpus -= current_produce_num_gpus
                if current_producer_stage != bottleneck:
                    partner_stages.append(current_producer_stage)
                    partner_actions.append(f"dec_{current_producer_dim}")
                if _consume_num_gpus > 0:
                    current_producer_stage, current_producer_dim = get_next_producer(config, partner_stages + [bottleneck], _consume_num_gpus)
                    debug_info(f"[DEBUG action_tp_dp] producer_stage: {current_producer_stage}, producer_dim: {current_producer_dim}", args.print_gpu_mig_details)
                    if current_producer_stage is None:
                        break
                    current_produce_num_gpus = config.stages[current_producer_stage].num_gpus // 2

        if check_legality(new_config, args):  
            best_config = new_config 
            debug_info(f"legal migration: partner stages {partner_stages} ({partner_actions})", args.print_gpu_mig_details)
            break
        else:
            debug_info(f"NOT legal migration: partner stages {partner_stages} ({partner_actions})", args.print_gpu_mig_details)

    if best_config is not None:
        update_recompute(best_config)
    return best_config    

def get_next_mbs(mbs, mbs_list, inc=True):
    assert mbs in mbs_list, f"{mbs} not in mbs_list"

    if inc:
        for i in range(len(mbs_list)):
            if mbs_list[i] == mbs:
                if i < len(mbs_list) - 1:
                    return mbs_list[i+1]
                else:
                    return mbs
    else:
        for i in range(len(mbs_list)):
            if mbs_list[i] == mbs:
                if i > 0:
                    return mbs_list[i-1]
                else:
                    return mbs   

def best_total_gpu_time(stage_config, base_batch_size, num_gpus):
    """ Given a stage config and usable # of GPUs, 
        output the best time and parallelism strategy for this stage """

    if(aceso_var.args.no_dp):
        return 1 , stage_config
    ops = stage_config.ops
    num_stages_behind = stage_config.num_stages_behind
    node_id_num = stage_config.node_id_num
    if num_gpus == 0:
        return MAX_VALUE, None

    best_time = MAX_VALUE
    best_stage_config = copy.deepcopy(stage_config)
    tp_size = 1
    cp_size = 1
    # while tp_size <= args.max_tp:
    while tp_size <= get_max_tp(aceso_var.args,stage_config):
        cp_size = 1
        while cp_size <= aceso_var.args.max_cp:
            dp_size = num_gpus // tp_size // cp_size
            if dp_size >= 1 and base_batch_size//dp_size in aceso_var.args.micro_batch_size:
                _tp_size = [tp_size for _ in range(len(ops))]
                _dp_size = [dp_size for _ in range(len(ops))]
                _cp_size = [cp_size for _ in range(len(ops))]
                recompute_ops = check_recompute(node_id_num,ops, base_batch_size, _tp_size, _dp_size , _cp_size, num_stages_behind, stage_config.algo)
                total_gpu_time = predict_stage_time(node_id_num,ops, recompute_ops, _tp_size, _dp_size, _cp_size, base_batch_size, stage_config.algo) * num_gpus
                # memory penalty
                stage_memory = predict_stage_memory(ops, recompute_ops, _tp_size, _dp_size, _cp_size, base_batch_size, num_stages_behind, stage_config.algo)
                # print("stage_memory",stage_memory)
                # if stage_memory > args.memory_limit:
                if stage_memory > get_memory_limit(aceso_var.args,stage_config.node_id_num):
                    total_gpu_time += stage_memory * 10

                if total_gpu_time < best_time:
                    best_time = total_gpu_time
                    best_stage_config.tp_size = _tp_size
                    best_stage_config.dp_size = _dp_size
                    best_stage_config.cp_size = _cp_size
                    best_stage_config.num_gpus = num_gpus
                    best_stage_config.recompute_ops = recompute_ops
            cp_size *=2
        tp_size *= 2

    if best_time < MAX_VALUE:
        return best_time, best_stage_config
    else:
        return best_time, None

def prim_mbs(config, bottleneck, action,args):
    global NUM_EXPLORED_CONFIGS
    NUM_EXPLORED_CONFIGS += 1
    base_batch_size = config.micro_bs
    if action == "inc_mbs":
        if base_batch_size < args.micro_batch_size[-1]:
            new_config_inc = copy.deepcopy(config)
            if(new_config_inc.micro_bs not in args.micro_batch_size or new_config_inc.micro_bs*2 not in args.micro_batch_size ):
                return None
            new_config_inc.micro_bs = get_next_mbs(new_config_inc.micro_bs, args.micro_batch_size, inc=True)
            for i in range(new_config_inc.num_stages):
                if not args.simple_prim_mbs:
                    _, new_stage_config = best_total_gpu_time(new_config_inc.stages[i], new_config_inc.micro_bs, new_config_inc.stages[i].num_gpus)               
                    if new_stage_config is not None:
                        new_config_inc.stages[i] = new_stage_config
                    else:
                        debug_info(f"no solution in (inc) mbs {new_config_inc.micro_bs}", args.print_debug_info)
                        return None
            return new_config_inc
        else:
            return None

    elif action == "dec_mbs":
        if base_batch_size > args.micro_batch_size[0]:
            new_config_dec = copy.deepcopy(config)
            if(new_config_dec.micro_bs not in args.micro_batch_size or new_config_dec.micro_bs//2 not in args.micro_batch_size):
                return None
            new_config_dec.micro_bs = get_next_mbs(new_config_dec.micro_bs, args.micro_batch_size, inc=False)
            for i in range(new_config_dec.num_stages):
                if not args.simple_prim_mbs:
                    _, new_stage_config = best_total_gpu_time(new_config_dec.stages[i], new_config_dec.micro_bs, new_config_dec.stages[i].num_gpus)
                    if new_stage_config is not None:
                        new_config_dec.stages[i] = new_stage_config
                    else:
                        return None
                else:
                    if new_config_dec[stage_name]["base_bs"] // max(new_config_dec[stage_name]["dp_size"]) not in args.micro_batch_size:
                        return None         
            return new_config_dec
        else:
            return None

def finetune_dim_stage_level(config):
    global NUM_EXPLORED_CONFIGS
    NUM_EXPLORED_CONFIGS += 1
    base_batch_size = config.micro_bs
    new_config_inc = copy.deepcopy(config)
    for i in range(new_config_inc.num_stages):
        _, new_stage_config = best_total_gpu_time(new_config_inc.stages[i], base_batch_size, new_config_inc.stages[i].num_gpus)               
        if new_stage_config is not None:
            new_config_inc.stages[i] = new_stage_config

    return new_config_inc

def finetune_dim_op_level_helper(config, index, op_index, inc_dim, reverse=False):
    """
    Helper function for finetune_dim_op_level
    """
    global NUM_EXPLORED_CONFIGS
    new_tp_size, new_dp_size, new_recompute_ops, new_time, new_memory = None, None, None, None, None 

    ops = config.stages[index].ops
    tp_size = config.stages[index].tp_size
    dp_size = config.stages[index].dp_size
    base_batch_size = config.micro_bs
    node_id_num = config.stages[index].node_id_num
    if reverse:
        _range = range(op_index, -1, -1)
    else:
        _range = range(op_index, len(ops))

    if inc_dim == "dp":
        for _op_index in _range:
            if tp_size[_op_index] > 1 and base_batch_size//(dp_size[_op_index]*2) in aceso_var.args.micro_batch_size:
                if new_tp_size is None:
                    new_tp_size = list(tp_size)
                    new_dp_size = list(dp_size)
                    NUM_EXPLORED_CONFIGS += 1
                new_tp_size[_op_index] //= 2
                new_dp_size[_op_index] *= 2    
            else:
                break
    elif inc_dim == "tp":
        for _op_index in _range:
            # if tp_size[_op_index] < args.max_tp and dp_size[_op_index] > 1 and \
            #     base_batch_size//(dp_size[_op_index]//2) in args.micro_batch_size:
            if tp_size[_op_index] < get_max_tp(aceso_var.args,config.stages[index]) and dp_size[_op_index] > 1 and \
                base_batch_size//(dp_size[_op_index]//2) in aceso_var.args.micro_batch_size:
                if new_tp_size is None:
                    new_tp_size = list(tp_size)
                    new_dp_size = list(dp_size)
                    NUM_EXPLORED_CONFIGS += 1
                new_tp_size[_op_index] *= 2
                new_dp_size[_op_index] //= 2    
            else:
                break 
    else:
        raise RuntimeError(f"inc dim {inc_dim} not supported.")

    if new_tp_size is not None:
        new_recompute_ops = check_recompute(node_id_num,ops, base_batch_size, new_tp_size, new_dp_size, config.stages[index].num_stages_behind, config.stages[index].algo)
        new_time = predict_stage_time(node_id_num,ops, new_recompute_ops, new_tp_size, new_dp_size, base_batch_size, config.stages[index].algo)
        new_memory = predict_stage_memory(ops, new_recompute_ops, new_tp_size, new_dp_size, base_batch_size, config.stages[index].num_stages_behind, config.stages[index].algo)

    return new_tp_size, new_dp_size, new_recompute_ops, new_time, new_memory

def finetune_dim_op_level(config, index, goal="time"):
    """
    Check the performance of modifying tp/dp of a sequence of consecutive operators,
    and return the best one. Starting from each op, we keep the the longest sequence from each direction (left to right, or right to left). 
    """
    global NUM_EXPLORED_CONFIGS

    if config.stages[index].num_gpus == 1:
        return None

    ops = config.stages[index].ops
    tp_size = config.stages[index].tp_size
    dp_size = config.stages[index].dp_size
    base_batch_size = config.micro_bs
    algo_list = config.stages[index].algo
    num_stages_behind = config.stages[index].num_stages_behind

    time_list = []
    memory_list = []
    tuned_tp_size_list = []
    tuned_dp_size_list = []
    tuned_recompute_ops_list = []

    for start_op_index in range(len(ops)):
        if ops[start_op_index] in ops_tunable:
            for inc_dim in ["tp", "dp"]:
                for reverse in [False, True]:
                    new_tp_size, new_dp_size, new_recompute_ops, new_time, new_memory \
                        = finetune_dim_op_level_helper(config, index, start_op_index, inc_dim, reverse=reverse)
                    if new_tp_size is not None:
                        tuned_tp_size_list.append(new_tp_size)
                        tuned_dp_size_list.append(new_dp_size)
                        tuned_recompute_ops_list.append(new_recompute_ops)
                        time_list.append(new_time)
                        memory_list.append(new_memory)

    if len(tuned_tp_size_list) > 0:
        best_index = 0   
        # if goal == "time" and min(memory_list) < args.memory_limit:  
        if goal == "time" and min(memory_list) < get_memory_limit(aceso_var.args,config.stages[index].node_id_num):
            best_time = MAX_VALUE
            for i in range(len(tuned_tp_size_list)):
                # if memory_list[i] < args.memory_limit and time_list[i] < best_time:
                if memory_list[i] < get_memory_limit(aceso_var.args,config.stages[index].node_id_num) and time_list[i] < best_time:
                    best_time = time_list[i]
                    best_index = i
        else:
            best_memory = MAX_VALUE
            for i in range(len(tuned_tp_size_list)):
                if memory_list[i] < best_memory:
                    best_memory = memory_list[i]
                    best_index = i            

        new_config = copy.deepcopy(config)
        new_config.stages[index].tp_size = tuned_tp_size_list[best_index]
        new_config.stages[index].dp_size = tuned_dp_size_list[best_index]
        new_config.stages[index].recompute_ops = tuned_recompute_ops_list[best_index]

        return new_config
    else:
        return None     

def finetune_algo_op_level(config, index):
    global NUM_EXPLORED_CONFIGS

    if config.stages[index].num_gpus == 1:
        return None

    ops = config.stages[index].ops
    recompute_ops = config.stages[index].recompute_ops
    tp_size = config.stages[index].tp_size
    dp_size = config.stages[index].dp_size
    base_batch_size = config.micro_bs
    algo_list = config.stages[index].algo
    num_stages_behind = config.stages[index].num_stages_behind
    node_id_num = config.stages[index].node_id_num
    prev_time = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list)
    prev_memory = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list)

    action_success = False 
    for i in range(len(ops)):
        if ops[i] in ops_tunable:
            time_list = []
            memory_list = []
            new_algos_list = []
            for algo_index in range(aceso_var.args.num_algos):
                if algo_index != algo_list[i]:
                    NUM_EXPLORED_CONFIGS += 1
                    new_algos = list(algo_list)
                    new_algos[i] = algo_index   
                    time_list.append(predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, new_algos,node_id_num))
                    memory_list.append(predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, new_algos))  
                    new_algos_list.append(new_algos)         

            best_index = -1
            # if prev_memory > args.memory_limit:
            if prev_memory > get_memory_limit(aceso_var.args,node_id_num):
                best_time = MAX_VALUE
                for j in range(len(time_list)):
                    if time_list[j] < best_time:
                        best_time = time_list[j]
                        best_index = j
            else:
                best_time = prev_time
                for j in range(len(time_list)):
                    # if time_list[j] < best_time and memory_list[j] < args.memory_limit:
                    if time_list[j] < best_time and memory_list[j] < get_memory_limit(aceso_var.args,node_id_num):
                        best_time = time_list[j]
                        best_index = j

            if best_index >= 0:
                action_success = True
                algo_list = new_algos_list[best_index]
                prev_memory = memory_list[best_index]
                prev_time = time_list[best_index]

    if action_success:
        new_config = copy.deepcopy(config)
        new_config.stages[index].algo = algo_list
        return new_config
    else:
        return None

def finetune(config,args=None):
    for stage_index in range(config.num_stages):
        initial_time = config.time_list[stage_index]
        config_ = finetune_dim_op_level(config, stage_index)
        if config_ is not None:
            predict_time_breakdown(config_,args=args)
            time_ = config_.time_list[stage_index]
            memory_ = config_.memory_list[stage_index]      
            # if time_ < initial_time and memory_ <= args.memory_limit:
            if time_ < initial_time and memory_ <= get_memory_limit(args,config.stages[stage_index].node_id_num):
                config = config_ 
                initial_time = time_ 

        config_ = finetune_algo_op_level(config, stage_index)
        if config_ is not None:
            predict_time_breakdown(config_,args=args)  
            time_ = config_.time_list[stage_index]
            memory_ = config_.memory_list[stage_index]                       
            # if time_ < initial_time and memory_ <= args.memory_limit:
            if time_ < initial_time and memory_ <= get_memory_limit(args,config.stages[stage_index].node_id_num):
                config = config_ 

    return config

def prim_tp_dp_exchange(config, index, action,args):
    global NUM_EXPLORED_CONFIGS
    NUM_EXPLORED_CONFIGS += 1

    new_config = None
    tp_size = config.stages[index].tp_size
    dp_size = config.stages[index].dp_size
    base_batch_size = config.micro_bs

    # if action == "inc_tp_dec_dp" and min(dp_size) > 1 and max(tp_size) * 2 <= args.max_tp:
    if action == "inc_tp_dec_dp" and min(dp_size) > 1 and max(tp_size) * 2 <= get_max_tp(args,config.stages[index]):
        new_config = copy.deepcopy(config)
        for i in range(len(new_config.stages[index].ops)):
            new_config.stages[index].tp_size[i] *= 2
            new_config.stages[index].dp_size[i] //= 2
    elif action == "inc_dp_dec_tp" and min(tp_size) > 1:
        for i in range(len(dp_size)):
            if base_batch_size // (dp_size[i] * 2) not in args.micro_batch_size:
                return None
        new_config = copy.deepcopy(config)
        for i in range(len(new_config.stages[index].ops)):
            new_config.stages[index].dp_size[i] *= 2
            new_config.stages[index].tp_size[i] //= 2

    if new_config is not None:
        update_recompute(new_config)     
    
    return new_config 

def get_target_parallelism(config: AcesoConfig,index,action):


    tp_size = config.stages[index].tp_size[0]
    dp_size = config.stages[index].dp_size[0]
    cp_size = config.stages[index].cp_size[0]
    base_batch_size = config.micro_bs
    gpu_type =list( config.stages[index].node_id_num.keys())[0]
    ops = config.stages[index].ops
    num_stages_behind = config.stages[index].num_stages_behind
    algo = config.stages[index].algo[0]
    compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd ,reserved_bwd = aceso_var.compute_fwd_time, aceso_var.compute_bwd_time, aceso_var.input_size, aceso_var.output_size, aceso_var.weights, aceso_var.activations, aceso_var.reserved_fwd, aceso_var.reserved_bwd
    device_num = dp_size * tp_size * cp_size

    execute_time = {}
    execute_memory = {}
    time_memory_pair = {}
    execute_weight = {}
    execute_activations= {}
    execute_fwd_reserved= {}
    execute_bwd_reserved= {}
    for op_name in compute_fwd_time[gpu_type].keys():
        for mbs_index in range(len(compute_fwd_time[gpu_type][op_name])):
            if(op_name in ops):
                if(math.pow(2,mbs_index)> base_batch_size):
                    break
                dp = base_batch_size//math.pow(2,mbs_index)
                for tp_index in range(len(compute_fwd_time[gpu_type][op_name][mbs_index])):
                    tp = math.pow(2,tp_index)
                    for cp_index in range(len(compute_fwd_time[gpu_type][op_name][mbs_index][tp_index])):
                        cp = math.pow(2,cp_index)
                        if(dp*tp*cp==device_num):
                            op_count = ops.count(op_name)
                            if((dp,tp,cp) not in execute_time):
                                execute_time[(dp,tp,cp)] = 0
                                execute_memory[(dp,tp,cp)] = 0
                                execute_activations[(dp,tp,cp)] = 0
                                execute_fwd_reserved[(dp,tp,cp)] = 0
                                execute_bwd_reserved[(dp,tp,cp)] = 0 
                                execute_weight[(dp,tp,cp)] = 0 
                            execute_time[(dp,tp,cp)] += compute_fwd_time[gpu_type][op_name][mbs_index][tp_index][cp_index][algo] * op_count
                            execute_time[(dp,tp,cp)] += compute_bwd_time[gpu_type][op_name][mbs_index][tp_index][cp_index][algo] * op_count
                            execute_weight[(dp,tp,cp)]+= weights[op_name][mbs_index][tp_index][cp_index][algo] * op_count
                            if aceso_var.args.consider_shared_space and op_name == "enc-attention-dropout":
                                execute_activations[(dp,tp,cp)] += activations[op_name][mbs_index][tp_index][cp_index][algo]*1.5 * op_count
                            elif aceso_var.args.consider_shared_space and (op_name in ["enc-attention-softmax", "bn1"] or "-bn3" in op_name or ("-downsample" in op_name and "0-0" not in op_name)):
                                execute_activations[(dp,tp,cp)] += 0
                            elif op_name == "encoder-embedding":
                                execute_activations[(dp,tp,cp)] += 0
                            else:
                                execute_activations[(dp,tp,cp)] += activations[op_name][mbs_index][tp_index][cp_index][algo]
                            execute_fwd_reserved[(dp,tp,cp)] = max(execute_fwd_reserved[(dp,tp,cp)], reserved_fwd[op_name][mbs_index][tp_index][cp_index][algo])
                            execute_bwd_reserved[(dp,tp,cp)] = max(execute_bwd_reserved[(dp,tp,cp)], reserved_bwd[op_name][mbs_index][tp_index][cp_index][algo])

    for key in execute_time.keys():
        execute_memory[key] += execute_weight[key] * (aceso_var.args.memory_main_params * aceso_var.args.memory_optimizer + 2)
        execute_memory[key] += max(execute_fwd_reserved[key] + execute_bwd_reserved[key], execute_weight[key]) 
        execute_memory[key] += execute_activations[key] * num_stages_behind
        time_memory_pair[key] = (execute_time[key], execute_memory[key])
    
    sort_parallel_time_memory = sorted(time_memory_pair.keys(),
                                       key=lambda x: time_memory_pair[x][0])

    new_parallel_time_memory = []
    last_item_memory = 1e8
    for i in sort_parallel_time_memory:
        current_memory = time_memory_pair[i][1]
        if (current_memory < last_item_memory):
            new_parallel_time_memory.append(i)
            last_item_memory = current_memory
    # for i in range(len(new_parallel_time_memory)):
    #     print("new_parallel_time_memory",new_parallel_time_memory[i], execute_time[new_parallel_time_memory[i]],execute_memory[new_parallel_time_memory[i]])

    
    # for i in time_memory_pair.keys():
    #     print("time_memory_pair",i, time_memory_pair[i][0],time_memory_pair[i][1])

    
    # exit()
    if(action=="dec_memory"):
        if((dp_size,tp_size,cp_size) in new_parallel_time_memory):
            idx = new_parallel_time_memory.index((dp_size,tp_size,cp_size))
            if(idx+1 < len(new_parallel_time_memory)):
                return new_parallel_time_memory[idx+1][0],new_parallel_time_memory[idx+1][1],new_parallel_time_memory[idx+1][2]
            else:
                return None,None ,None
        else:
            return new_parallel_time_memory[-1][0],new_parallel_time_memory[-1][1],new_parallel_time_memory[-1][2]
            


    else:
        if((dp_size,tp_size,cp_size) in new_parallel_time_memory):
            idx = new_parallel_time_memory.index((dp_size,tp_size,cp_size))
            if(idx> 0):
                return new_parallel_time_memory[idx-1][0],new_parallel_time_memory[idx-1][1],new_parallel_time_memory[idx-1][2]
            else:
                return None ,None ,None
        else:
            return new_parallel_time_memory[0][0],new_parallel_time_memory[0][1],new_parallel_time_memory[0][2]


    return None

def prim_tp_dp_cp_exchange(config: AcesoConfig, index, action, args):
    global NUM_EXPLORED_CONFIGS
    NUM_EXPLORED_CONFIGS += 1

    new_config = None
    new_config = copy.deepcopy(config)

    
    target_dp, target_tp, target_cp = get_target_parallelism(config,index,action)
    if(target_dp == None):
        return None
    for i in range(len(new_config.stages[index].ops)):
        new_config.stages[index].dp_size[i] = int(target_dp)
        new_config.stages[index].tp_size[i] = int(target_tp)
        new_config.stages[index].cp_size[i] = int(target_cp)
    

    if new_config is not None:
        update_recompute(new_config)

    return new_config


action_resource_table = [
    AcesoPrim(name = "dec_op",  time = "-", memory = "-", num_devices = "0", workloads = "-", efficiency = "0", comm = "0", func = prim_mig_op if aceso_var.args.op_grain == "operator" else (prim_mig_op_block if aceso_var.args.op_grain == "block" else (prim_mig_op_layer if aceso_var.args.op_grain == "layer" else prim_mig_op_simple))),
    AcesoPrim(name = "inc_mbs", time = "-", memory = "+", num_devices = "0", workloads = "0", efficiency = "+", comm = "0", func = prim_mbs),
    AcesoPrim(name = "dec_mbs", time = "+", memory = "-", num_devices = "0", workloads = "0", efficiency = "-", comm = "0", func = prim_mbs)
]

if aceso_var.args.add_action_tp_dp_exchange and not aceso_var.args.no_dp:
    action_resource_table += [
        AcesoPrim(name = "inc_tp_dec_dp", time = "+", memory = "-", num_devices = "0", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp_exchange),
        AcesoPrim(name = "inc_dp_dec_tp", time = "-", memory = "+", num_devices = "0", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp_exchange)
    ]
if(aceso_var.args.add_action_tp_dp and not aceso_var.args.no_dp):
    action_resource_table += [
    AcesoPrim(name = "inc_dp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    AcesoPrim(name = "dec_dp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    AcesoPrim(name = "inc_tp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    AcesoPrim(name = "dec_tp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    ]
if(aceso_var.args.add_action_tp_cp_dp_exchange):
    action_resource_table += [
        AcesoPrim(name = "dec_memory",  time = "+", memory = "-", num_devices = "0", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp_cp_exchange),
        AcesoPrim(name = "dec_time",  time = "-", memory = "+", num_devices = "0", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp_cp_exchange),
    ]
OP_SUCCESS_LINEAR_COUNT=0
OP_SUCCESS_BISECTION_COUNT=0
OP_SUCCESS_LINEAR_COUNT_LIST=[]
OP_SUCCESS_BISECTION_COUNT_LIST=[]

import numpy as np
def get_op_success_linear_count():
    return OP_SUCCESS_LINEAR_COUNT_LIST
def get_mean_op_success_linear_count():
    # print(OP_SUCCESS_LINEAR_COUNT_LIST)
    if(len(OP_SUCCESS_LINEAR_COUNT_LIST)==0):
        return 0
    return np.mean(list(OP_SUCCESS_LINEAR_COUNT_LIST))
def get_op_success_bisection_count():
    return OP_SUCCESS_BISECTION_COUNT_LIST
def get_mean_op_success_bisection_count():
    # print(OP_SUCCESS_BISECTION_COUNT_LIST)
    if(len(OP_SUCCESS_BISECTION_COUNT_LIST)==0):
        return 0
    return np.mean(list(OP_SUCCESS_BISECTION_COUNT_LIST))