# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/training.py
# Git commit hash: 42c1cf4279acea5a554500dcb552211f44cbec45
# We retain the following copyright from the original files:

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
# print(f"[DEBUG_TIME] at the beginning: _TRAIN_START_TIME = {_TRAIN_START_TIME}")
import torch
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_memory_record
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_without_interleaving_simi
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.utils import report_memory
from megatron.mpu.utils import ensure_divisibility
from torch.profiler import profile, record_function, ProfilerActivity
from megatron.prefix_match import process_in_order,partition_micro_batch,kk_partition,get_store_shared_tensor, partition_micro_batch_token_level
import torch.cuda.nvtx as nvtx
import csv
import os
import gc

DEBUG_GRAD = os.environ.get("DEBUG_GRAD", '0') == '1'
DEBUG_FIX_WEIGHT = os.environ.get("DEBUG_FIX_WEIGHT", '0') == '1'
DEBUG_COMMUNICATE = os.environ.get("DEBUG_COMMUNICATE", '0') == '1'
ENABLE_WEIGHT_SHARE = os.environ.get("ENABLE_WEIGHT_SHARE", '1') == '1'

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    print(f"[rank {torch.distributed.get_rank()}]")
    torch.distributed.barrier()
    print(f"[rank {torch.distributed.get_rank()}] sync")
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))





def initialize_data():
    args = get_args()
    tokenized_data = torch.load(args.input_token_path)
    current_pipe = args.current_pipe
    pipe_num = args.pipe_num * args.dp_simi

    data = []

    for uid in tokenized_data.keys():
        for tracj_id in tokenized_data[uid].keys():
            for turn in tokenized_data[uid][tracj_id].keys():
                tokenized_data[uid][tracj_id][turn] = tokenized_data[uid][tracj_id][turn]
                data.append(tokenized_data[uid][tracj_id][turn])

    # 打乱 data
    np.random.seed(43)
    np.random.shuffle(data)


    distributed_data = []
    for i in range(len(data)):
        if i % pipe_num == current_pipe:
            distributed_data.append(data[i])
    
    print("len of distributed data for pipe ", current_pipe, " is ", len(distributed_data))
    # mbs = args.micro_batch_size
    gbs = len(distributed_data)+1  # global batch size

    if(args.enable_token_level_balance):
        part_mbs, cache_ratio_list = partition_micro_batch_token_level(distributed_data, partition_n=100,max_token=args.max_token_num)
    else:
        np.random.seed(44)
        np.random.shuffle(distributed_data)
        part_mbs, cache_ratio_list = partition_micro_batch(distributed_data, partition_n=100,max_token=args.max_token_num)

    args.train_iters= 1

    args.train_iters += args.system_warm 
    print("Args train iters after adding system warmup: ", args.train_iters)
    args.nbs_each_iter = {}
    args.gbs_each_iter = []
    for i in range(args.system_warm):
        args.gbs_each_iter.append(gbs)
        args.nbs_each_iter[i] = gbs // 4

    collective_mbs_data = []
    args.mbs_each_forward =  [ len(mb) for mb in part_mbs]

    last_forward_id = 0
    forward_id = 0
    current_sample = 0
    for i in range(args.system_warm,args.train_iters):
        if (i+1-args.system_warm)*gbs <= len(distributed_data):
            args.gbs_each_iter.append(gbs)
        else:
            args.gbs_each_iter.append(len(distributed_data) - (i-args.system_warm) * gbs)
        while(current_sample < args.gbs_each_iter[-1]):
            if forward_id >= len(args.mbs_each_forward) or current_sample + args.mbs_each_forward[forward_id] >= args.gbs_each_iter[-1]:
                break
            current_sample += args.mbs_each_forward[forward_id]
            forward_id += 1
        args.nbs_each_iter[i] = forward_id - last_forward_id
        last_forward_id = forward_id
        current_sample = 0

    last_idx = 0
    args.shared_prefix_len = []
    args.store_for_sample_idx = []
    args.shared_for_sample_idx = []
    args.cache_ratio_list = [0.0] * len(args.mbs_each_forward)
    args.batch_idx_mapping_sample_idx = []


    if(args.intra_batch_share != True):
        shared_prefix_len, store_for_sample_idx, shared_for_sample_idx = get_store_shared_tensor(distributed_data)
        for i in args.mbs_each_forward:
            start_idx = last_idx
            end_idx = min(last_idx + i, len(distributed_data))
            # 如果shared 长度等于该micro batch长度，则说明该micro batch不需要计算，将它剔除
            if sum(shared_prefix_len[start_idx:end_idx]) == sum([ len(distributed_data[j]) for j in range(start_idx, end_idx)]):
                args.nbs_each_iter[args.train_iters -1] -= 1
                # print("remove last micro batch ", start_idx, end_idx, " from the last iteration")
                last_idx = end_idx
                continue
            collective_mbs_data.append(distributed_data[start_idx:end_idx])
            args.shared_prefix_len.append(shared_prefix_len[start_idx:end_idx])
            args.store_for_sample_idx.append(store_for_sample_idx[start_idx:end_idx])
            args.shared_for_sample_idx.append(shared_for_sample_idx[start_idx:end_idx])
            args.batch_idx_mapping_sample_idx.append((start_idx, end_idx))
            last_idx = end_idx
    else:
        for i in args.mbs_each_forward:
            start_idx = last_idx
            end_idx = min(last_idx + i, len(distributed_data))
            shared_prefix_len, store_for_sample_idx, shared_for_sample_idx = get_store_shared_tensor(distributed_data[start_idx:end_idx])
            collective_mbs_data.append(distributed_data[start_idx:end_idx])
            args.shared_prefix_len.append(shared_prefix_len)
            args.store_for_sample_idx.append(store_for_sample_idx)
            args.shared_for_sample_idx.append(shared_for_sample_idx)
            args.batch_idx_mapping_sample_idx.append((start_idx, end_idx))
            last_idx = end_idx
    args.distributed_data = collective_mbs_data
    # print("args.shared_prefix_len",args.shared_prefix_len)
    cache_token = sum([sum(micro_batch_cache_time) for micro_batch_cache_time in args.shared_prefix_len])
    total_token = sum([len(d) for d in distributed_data])
    print("pipeline", current_pipe, ": total tokens ", total_token, "cache token ", cache_token, "compute token ", total_token - cache_token, "cache ratio ", 1 - (total_token - cache_token)/ total_token)
    print("train iters = ", args.train_iters, "nbs_each_iter" ,args.nbs_each_iter )




def balance_initialize_data():
    args = get_args()
    tokenized_data = torch.load(args.input_token_path)
    current_pipe = args.current_pipe
    pipe_num = args.pipe_num * args.dp_simi

    total_data = []
    total_tokens = 0
    uid_data_order = []
    uid_data_comp_count = []
    global_len = 0
    partition_n = pipe_num
    part_id  = current_pipe
    for uid in tokenized_data.keys():
        uid_data = []
        for tracj_id in tokenized_data[uid].keys():
            if(type(tokenized_data[uid][tracj_id]) == list and len(tokenized_data[uid][tracj_id]) >0 and type(tokenized_data[uid][tracj_id][0]) == list): # Tree GRPO
                for turn in range(len(tokenized_data[uid][tracj_id])):
                    total_data.append(tokenized_data[uid][tracj_id][turn])
                    uid_data.append(tokenized_data[uid][tracj_id][turn])
                    global_len += len(tokenized_data[uid][tracj_id][turn])
            elif(type(tokenized_data[uid][tracj_id]) == list): # Tree Step
                total_data.append(tokenized_data[uid][tracj_id])
                uid_data.append(tokenized_data[uid][tracj_id])
                global_len += len(tokenized_data[uid][tracj_id])
            elif(type(tokenized_data[uid][tracj_id]) == dict): # Step
                for turn in tokenized_data[uid][tracj_id].keys():
                    if(args.token_mode and int(turn) < len(tokenized_data[uid][tracj_id]) -1):
                        continue
                    total_data.append(tokenized_data[uid][tracj_id][turn])
                    uid_data.append(tokenized_data[uid][tracj_id][turn])
                    global_len += len(tokenized_data[uid][tracj_id][turn])
            else:
                raise ValueError("tokenized_data format error")
        uid_data_comp_count.append(process_in_order(uid_data))
        uid_data_order.append(uid)

    global_compu_count = process_in_order(total_data)
    print("global len ", global_len, " global compu count ", global_compu_count, "cache ratio ", 1- global_compu_count / global_len)
    
    distributed_data = []
    if(args.dp_workload_balance):
        kk_partitions, kk_sums, kk_order = kk_partition(uid_data_comp_count, uid_data_order, partition_n)
        print("pipe ", current_pipe, " kk sums ", kk_sums)
        # print(f"Processing partition {part_idx} with Karmarkar-Karp assignment")
        part_uids = [kk_order[part_id][i] for i in range(len(kk_order[part_id])) if uid_data_comp_count[uid_data_order.index(kk_order[part_id][i])] in kk_partitions[part_id]]
        # print("pipe ", current_pipe, " get kk partition uids ", part_uids)
        for uid in part_uids:
            for tracj_id in tokenized_data[uid].keys():
                if(type(tokenized_data[uid][tracj_id]) == list and len(tokenized_data[uid][tracj_id]) >0 and type(tokenized_data[uid][tracj_id][0]) == list):
                    for turn in range(len(tokenized_data[uid][tracj_id])):
                        distributed_data.append(tokenized_data[uid][tracj_id][turn])
                        total_tokens += len(tokenized_data[uid][tracj_id][turn])
                elif(type(tokenized_data[uid][tracj_id])== list):
                    distributed_data.append(tokenized_data[uid][tracj_id])
                    total_tokens += len(tokenized_data[uid][tracj_id])
                elif(type(tokenized_data[uid][tracj_id]) == dict):
                    for turn in tokenized_data[uid][tracj_id].keys():
                        if(args.token_mode and int(turn) < len(tokenized_data[uid][tracj_id]) -1):
                            continue
                        distributed_data.append(tokenized_data[uid][tracj_id][turn])
                        total_tokens += len(tokenized_data[uid][tracj_id][turn])
                else:
                    raise ValueError("tokenized_data format error")
    else:
        np.random.seed(43)
        np.random.shuffle(total_data)
        for i in range(len(total_data)):
            if i % pipe_num == current_pipe:
                distributed_data.append(total_data[i])
                total_tokens += len(total_data[i])
    count = process_in_order(distributed_data)
    
    print("len of distributed data for pipe ", current_pipe, " is ", len(distributed_data))
    # mbs = args.micro_batch_size
    gbs = len(distributed_data)+1  # global batch size

    if(args.enable_token_level_balance):
        part_mbs = partition_micro_batch_token_level(distributed_data, partition_n=100,max_token=args.max_token_num)
    else:
        np.random.seed(44)
        np.random.shuffle(distributed_data)
        part_mbs = partition_micro_batch(distributed_data, partition_n=100, max_token=args.max_token_num)

    args.train_iters= 1

    args.train_iters += args.system_warm 
    print("Args train iters after adding system warmup: ", args.train_iters)
    args.nbs_each_iter = {}
    args.gbs_each_iter = []
    for i in range(args.system_warm):
        args.gbs_each_iter.append(gbs)
        args.nbs_each_iter[i] = 32

    collective_mbs_data = []
    args.mbs_each_forward =  [ len(mb) for mb in part_mbs]

    last_forward_id = 0
    forward_id = 0
    current_sample = 0

    # args.nbs_each_iter[args.system_warm] = len(part_mbs)
    for i in range(args.system_warm,args.train_iters):
        if (i+1-args.system_warm)*gbs <= len(distributed_data):
            args.gbs_each_iter.append(gbs)
        else:
            args.gbs_each_iter.append(len(distributed_data) - (i-args.system_warm) * gbs)
        while(current_sample < args.gbs_each_iter[-1]):
            if forward_id >= len(args.mbs_each_forward) or current_sample + args.mbs_each_forward[forward_id] >= args.gbs_each_iter[-1]:
                break
            current_sample += args.mbs_each_forward[forward_id]
            forward_id += 1
        args.nbs_each_iter[i] = forward_id - last_forward_id
        last_forward_id = forward_id
        current_sample = 0

    last_idx = 0
    args.shared_prefix_len = []
    args.store_for_sample_idx = []
    args.shared_for_sample_idx = []
    args.batch_idx_mapping_sample_idx = []

    shared_prefix_len, store_for_sample_idx, shared_for_sample_idx = get_store_shared_tensor(distributed_data)
    for i in args.mbs_each_forward:
        start_idx = last_idx
        end_idx = min(last_idx + i, len(distributed_data))
        if sum(shared_prefix_len[start_idx:end_idx]) == sum([ len(distributed_data[j]) for j in range(start_idx, end_idx)]):
            args.nbs_each_iter[args.train_iters -1] -= 1
            # print("remove last micro batch ", start_idx, end_idx, " from the last iteration")
            last_idx = end_idx
            continue
        collective_mbs_data.append(distributed_data[start_idx:end_idx])
        args.shared_prefix_len.append(shared_prefix_len[start_idx:end_idx])
        args.store_for_sample_idx.append(store_for_sample_idx[start_idx:end_idx])
        args.shared_for_sample_idx.append(shared_for_sample_idx[start_idx:end_idx])
        last_idx = end_idx
        args.batch_idx_mapping_sample_idx.append((start_idx, end_idx))

    print("args.shared_for_sample_idx:", args.shared_for_sample_idx[1])
    print("args.store_for_sample_idx:", args.store_for_sample_idx[1])
    args.distributed_data = collective_mbs_data
    cache_token = sum([sum(micro_batch_cache_time) for micro_batch_cache_time in args.shared_prefix_len])
    total_token = sum([len(d) for d in distributed_data])
    print("pipeline", current_pipe, ": total tokens ", total_token, "cache token ", cache_token, "compute token ", total_token - cache_token, "cache ratio ", 1 - (total_token - cache_token)/ total_token)
    print("train iters = ", args.train_iters, "nbs_each_iter" ,args.nbs_each_iter )
    print("warm mbs token len", [len(sample) for sample in args.distributed_data[0]])

def get_share_info():
    args = get_args()
    intra_cache_rate = []
    inter_cache_rate = []
    self_cache_rate = []
    for batch_idx, mbs in enumerate(args.mbs_each_forward):
        start_idx, end_idx = args.batch_idx_mapping_sample_idx[batch_idx]
        intra_cache = 0
        inter_cache = 0
        self_cache = 0
        for idx,dict in enumerate(args.shared_for_sample_idx[batch_idx]):
            if(idx==0):
                for shared_idx in dict.keys():
                    if(shared_idx == start_idx-1):
                        self_cache += dict[shared_idx][1] - dict[shared_idx][0]
                    else:
                        inter_cache += dict[shared_idx][1] - dict[shared_idx][0]
            else:
                for shared_idx in dict.keys():
                    if(shared_idx >= start_idx):
                        intra_cache += dict[shared_idx][1] - dict[shared_idx][0]
                    else:
                        inter_cache += dict[shared_idx][1] - dict[shared_idx][0]
        sum_shared_prefix_len = intra_cache + inter_cache + self_cache
        intra_cache_rate.append(intra_cache / (sum_shared_prefix_len + 1e-6))
        inter_cache_rate.append(inter_cache / (sum_shared_prefix_len + 1e-6))
        self_cache_rate.append(self_cache / (sum_shared_prefix_len + 1e-6))
    if(args.rank == 0 and args.record_path is not None and args.record_prefix is not None):
        if not os.path.exists(args.record_path):
            os.makedirs(args.record_path, exist_ok=True)
        with open(f"{args.record_path}/{args.record_prefix}_cache_rate_pipeline_{args.current_pipe}_rank_{torch.distributed.get_rank()}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["intra_cache_rate", "inter_cache_rate", "self_cache_rate"])
            for i in range(len(intra_cache_rate)):
                writer.writerow([intra_cache_rate[i], inter_cache_rate[i], self_cache_rate[i]])

    return intra_cache_rate, inter_cache_rate, self_cache_rate

def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    args = get_args()

    balance_initialize_data()
    # intra_cache_rate, Inter_cache_rate, self_cache_rate = get_share_info()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    if(args.distributed_backend == "gloo"):
        start_time_tensor = start_time_tensor.cpu()
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('[TIME] time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))

    timers = get_timers()
    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model-and-optimizer-setup').stop()

    print_rank_0('[TIME] after model, optimizer, and learning rate scheduler are built (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))

    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()
    if args.virtual_pipeline_model_parallel_size is not None and args.virtual_pipeline_model_parallel_size > 1:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()

    print_rank_0('[TIME] after dataloaders are built (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))    

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)

    print_rank_0('[TIME] after training is done (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))                              

    # if args.do_valid:
    #     prefix = 'the end of training for val data'
    #     evaluate_and_print_results(prefix, forward_step_func,
    #                                valid_data_iterator, model,
    #                                iteration, False)

    # if args.save and iteration != 0:
    #     save_checkpoint(iteration, model, optimizer, lr_scheduler)

    # if args.do_test:
    #     # Run on test data.
    #     prefix = 'the end of training for test data'
    #     evaluate_and_print_results(prefix, forward_step_func,
    #                                test_data_iterator, model,
    #                                0, True)

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))

def set_weight(model, val=0.01):
    for param in model.parameters():
        param.data.fill_(val)

def get_op_via_index(op_index, models):
    for model in models:
        model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
        for op in model.language_model.ops:
            if op.op_index == op_index:
                return op
    return None

def send_shared_tensors(op, models, grads=False):
    
    args = get_args()
    shared_tensor = op.get_shared_tensor(grads=grads)

    for key in sorted(shared_tensor):
        for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
            if not op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index]:
                recv_ranks = op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index]
                if len(recv_ranks) > 0:
                    send_ops = []
                    split_dim = op.shared_weights_info[key]["tp_split_dim"]

                    for recv_tp_groups in recv_ranks:
                        if len(recv_tp_groups) == 1:
                            tensor_list = [shared_tensor[key]]
                        else:
                            if split_dim != -1:
                                tensor_list = list(torch.chunk(shared_tensor[key], chunks=len(recv_tp_groups), dim=split_dim)) 
                            else:
                                tensor_list = []
                                for _ in range(len(recv_tp_groups)):
                                    tensor_list.append(shared_tensor[key])

                        for i in range(len(tensor_list)):
                            send_op = torch.distributed.P2POp(
                                torch.distributed.isend, tensor_list[i].contiguous(), recv_tp_groups[i])
                            send_ops.append(send_op) 

                            if DEBUG_COMMUNICATE:
                                current_rank = torch.distributed.get_rank()
                                string = f"(shared) rank {current_rank} send to {recv_tp_groups[i]} size = {list(tensor_list[i].size())}"
                                with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
                                    f.write(string+"\n")    

                    if len(send_ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(send_ops)
                        for req in reqs:
                            req.wait()
                        torch.cuda.synchronize()

def recv_shared_tensors(op, models, grads=False):
    args = get_args()

    recv_dict = {}
    shared_tensor = op.get_shared_tensor(grads=False)
    for key in sorted(shared_tensor):
        recv_dict[key] = []

    for key in sorted(shared_tensor):
        if key == "position_embeddings" and not grads:
            dtype = torch.float32
        else:
            dtype = args.params_dtype        
        for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
            if op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index]:
                src_op = get_op_via_index(op_index, models)
                recv_tensor = src_op.get_shared_tensor(grads=grads)
                recv_dict[key].append(recv_tensor[key])
            else:
                send_ranks = op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index]
                if len(send_ranks) > 0: 
                    recv_ops = []
                    tensor_list = []
                    receive_size = list(shared_tensor[key].size())
                    split_dim = op.shared_weights_info[key]["tp_split_dim"]
                    if split_dim != -1:
                        receive_size[split_dim] //= len(send_ranks[0])
                        
                    for send_tp_groups in send_ranks:
                        tmp_tensor_list = []
                        for _ in range(len(send_tp_groups)):
                            tmp_tensor_list.append(torch.empty(receive_size, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype))
                        for i in range(len(tmp_tensor_list)):
                            recv_op = torch.distributed.P2POp(
                                torch.distributed.irecv, tmp_tensor_list[i], send_tp_groups[i])
                            recv_ops.append(recv_op)
                            if DEBUG_COMMUNICATE:
                                current_rank = torch.distributed.get_rank()
                                string = f"(shared) rank {current_rank} recv from {send_tp_groups[i]} size = {list(tmp_tensor_list[i].size())}"
                                with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
                                    f.write(string+"\n")    
                        tensor_list.append(tmp_tensor_list)

                    if len(recv_ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(recv_ops)
                        for req in reqs:
                            req.wait()
                        torch.cuda.synchronize()

                    if split_dim != -1:
                        if len(tensor_list) == 1:
                            recv_dict[key].append(torch.cat(tensor_list[0], dim=split_dim))
                        else:
                            result_tensor = torch.sum(torch.stack([torch.cat(tensor_list[i], dim=split_dim) for i in range(len(tensor_list))]), dim=0)
                            recv_dict[key].append(result_tensor)
                    else:
                        if len(tensor_list) == 1:
                            recv_dict[key].append(tensor_list[0][0])
                        else:
                            result_tensor = torch.sum(torch.stack([tensor_list[i][0] for i in range(len(tensor_list))]), dim=0)
                            recv_dict[key].append(result_tensor)
    return recv_dict

def initialize_weights_sharing(models):
    if ENABLE_WEIGHT_SHARE:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()    
        rank = torch.distributed.get_rank()
        # initialize the ranks
        for model in models:
            model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
            for op in model.language_model.ops:
                if len(op.shared_weights_info) > 0:
                    for key in sorted(op.shared_weights_info):
                        op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"] = {}   
                        op.shared_weights_info[key]["sharing_weights_with_ranks"] = {}                      
                        if op.shared_weights_info[key]["root"]:
                            # calculate & store the destination ranks. 
                            for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
                                dest_pipeline_rank = mpu.get_pipeline_rank_via_op_index(op_index)
                                if dest_pipeline_rank == pipeline_rank:
                                    op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = True
                                else:
                                    op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = False

                                    ranks_in_send_stage = mpu.get_ranks_via_pipeline_stage(pipeline_rank)
                                    ranks_in_receive_stage = mpu.get_ranks_via_pipeline_stage(dest_pipeline_rank)
                                    num_ranks_in_send_stage = len(ranks_in_send_stage)
                                    num_ranks_in_receive_stage = len(ranks_in_receive_stage)

                                    tp_size, dp_size = mpu.get_op_tp_size(op.op_index), mpu.get_op_dp_size(op.op_index)
                                    tp_size_next, dp_size_next = mpu.get_op_tp_size(op_index), mpu.get_op_dp_size(op_index)

                                    for i in range(num_ranks_in_send_stage):
                                        if ranks_in_send_stage[i] == rank:
                                            dp_id = i // tp_size
                                            tp_id = i % tp_size

                                    next_dp_id = [dp_id]
                                    next_tp_id = [tp_id]

                                    if tp_size_next > tp_size:
                                        ratio = tp_size_next // tp_size
                                        next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)                                    
                                    if tp_size_next < tp_size:
                                        ratio = tp_size // tp_size_next
                                        next_tp_id = [tp_id // ratio]  
                                    if dp_size_next > dp_size:
                                        ratio = dp_size_next // dp_size
                                        next_dp_id = range(dp_id * ratio, (dp_id + 1)*ratio)                                      
                                    if dp_size_next < dp_size:
                                        ratio = dp_size // dp_size_next
                                        if dp_id % ratio == 0:
                                            next_dp_id = [dp_id // ratio] 
                                        else:
                                            next_dp_id = []

                                    op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index] = []
                                    if len(next_dp_id) > 0:
                                        for _dp_id in next_dp_id:
                                            tmp_list = []
                                            for _tp_id in next_tp_id:
                                                tmp_list.append(ranks_in_receive_stage[_dp_id * tp_size_next + _tp_id])
                                            op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index].append(list(tmp_list))
                        else:
                            assert len(op.shared_weights_info[key]["sharing_with_ops"]) == 1
                            op_index = op.shared_weights_info[key]["sharing_with_ops"][0]
                            src_pipeline_rank = mpu.get_pipeline_rank_via_op_index(op_index)
                            if src_pipeline_rank == pipeline_rank:
                                op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = True
                            else:
                                op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = False

                                ranks_in_send_stage = mpu.get_ranks_via_pipeline_stage(src_pipeline_rank)
                                ranks_in_receive_stage = mpu.get_ranks_via_pipeline_stage(pipeline_rank)
                                num_ranks_in_send_stage = len(ranks_in_send_stage)
                                num_ranks_in_receive_stage = len(ranks_in_receive_stage)

                                tp_size, dp_size = mpu.get_op_tp_size(op.op_index), mpu.get_op_dp_size(op.op_index)
                                tp_size_next, dp_size_next = mpu.get_op_tp_size(op_index), mpu.get_op_dp_size(op_index)

                                for i in range(num_ranks_in_receive_stage):
                                    if ranks_in_receive_stage[i] == rank:
                                        dp_id = i // tp_size
                                        tp_id = i % tp_size

                                next_dp_id = [dp_id]
                                next_tp_id = [tp_id]

                                if tp_size_next > tp_size:
                                    ratio = tp_size_next // tp_size
                                    next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)                                    
                                if tp_size_next < tp_size:
                                    ratio = tp_size // tp_size_next
                                    next_tp_id = [tp_id // ratio]  
                                if dp_size_next > dp_size:
                                    ratio = dp_size_next // dp_size
                                    next_dp_id = [dp_id * ratio]                                 
                                if dp_size_next < dp_size:
                                    ratio = dp_size // dp_size_next
                                    next_dp_id = [dp_id // ratio]   

                                op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index] = []

                                for _dp_id in next_dp_id:
                                    tmp_list = []
                                    for _tp_id in next_tp_id:
                                        tmp_list.append(ranks_in_send_stage[_dp_id * tp_size_next + _tp_id])
                                    op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index].append(list(tmp_list))

        # send & receive tensors
        for model in models:
            model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
            for op in model.language_model.ops:
                if len(op.shared_weights_info) > 0:
                    is_root = False 
                    for key in op.shared_weights_info:
                        if op.shared_weights_info[key]["root"]:
                            is_root = True
                    if is_root:
                        send_shared_tensors(op, models, grads=False)
                    else:
                        recv_tensor = recv_shared_tensors(op, models, grads=False)
                        op.set_shared_tensor(recv_tensor, grads=False)
        

def synchronize_shared_weights_grads(models):
    if ENABLE_WEIGHT_SHARE:
        for model in models:
            model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
            # two-phase to avoid deadlock
            # Phase 1: root: receive, sum up, send out
            #          workers: send
            for op in model.language_model.ops:
                if len(op.shared_weights_info) > 0:
                    is_root = False
                    for key in op.shared_weights_info:
                        if op.shared_weights_info[key]["root"]:
                            is_root = True                
                    if is_root:
                        grads_dict = {}
                        recv_grads_dict = recv_shared_tensors(op, models, grads=True)
                        current_grads_dict = op.get_shared_tensor(grads=True)
                        for key in sorted(op.shared_weights_info):
                            # receive grads from all sync-ops.
                            recv_grads = recv_grads_dict[key]
                            # sum up the grads from all sync-ops and this op.
                            current_grads = current_grads_dict[key]
                            recv_grads.append(current_grads)
                            grads_dict[key] = [sum(recv_grads)]               
                        op.set_shared_tensor(grads_dict, grads=True)                    
                        # send sum of grads back to all the sync-ops.                  
                        send_shared_tensors(op, models, grads=True)                   
                    else:
                        # send grads to root op. 
                        send_shared_tensors(op, models, grads=True)

            # Phase 2: workers: receive
            for op in model.language_model.ops:
                if len(op.shared_weights_info) > 0:
                    is_root = False
                    for key in op.shared_weights_info:
                        if op.shared_weights_info[key]["root"]:
                            is_root = True                  
                    if not is_root:               
                        # recv sum of grads.
                        recv_grads = recv_shared_tensors(op, models, grads=True)
                        # update grads.
                        op.set_shared_tensor(recv_grads, grads=True)

def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None \
           and args.virtual_pipeline_model_parallel_size > 1:
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        if DEBUG_FIX_WEIGHT:
            set_weight(model_module)            

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    initialize_weights_sharing(model)

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = [torchDDP(model_module, device_ids=[i], output_device=i,
                          process_group=mpu.get_data_parallel_group())
                 for model_module in model]
        return model

    if args.DDP_impl == 'local':
        model = [LocalDDP(model_module,
                          args.accumulate_allreduce_grads_in_fp32,
                          args.use_contiguous_buffers_in_ddp)
                 for model_module in model]
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    model = get_model(model_provider_func)
    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)

    lr_scheduler = get_learning_rate_scheduler(optimizer)
    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers('load-checkpoint').start()
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        torch.distributed.barrier()
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler,iteration):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
        for partition in model:
            partition.zero_grad_buffer()

    ## New Megatron
    if optimizer is not None:
        optimizer.zero_grad()    

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            if args.virtual_pipeline_model_parallel_size > 1:
                forward_backward_func = forward_backward_pipelining_with_interleaving
                assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                    'number of microbatches is not divisible by pipeline-parallel ' \
                    'size when using interleaved schedule'
            else:
                if(args.simi_pipe):
                    forward_backward_func = forward_backward_pipelining_without_interleaving_simi
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving

        else:
            if(args.simi_pipe):
                forward_backward_func = forward_backward_pipelining_without_interleaving_simi
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving

    else:
        forward_backward_func = forward_backward_no_pipelining

    nvtx.range_push("1F1B")
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=False, iteration=iteration)
    nvtx.range_pop()
    # Empty unused memory (From new Megatron)
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    nvtx.range_push("All-Reduce")
    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()
    nvtx.range_pop()
    if DEBUG_GRAD:
        string = f"=================== grad info BEFORE sync [rank {torch.distributed.get_rank()}] ==================="
        with open(f"{args.log_path}{args.log_name}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
            f.write(string+"\n")    
        total_size = 0
        for name, params in model[0].named_parameters():
            param_size = list(params.data.size())
            string = f"[DEBUG] param name {name}, grad_requires: {params.requires_grad},\n weight({params.data.dtype}): {params.data} \n grad_value ({params.main_grad.dtype}): {params.main_grad}"
            with open(f"{args.log_path}{args.log_name}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
                f.write(string+"\n")  
        print(f"[TOTAL PARAMS SIZE] {total_size} MB")

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    timers('backward-embedding-all-reduce').start()
    synchronize_shared_weights_grads(model)
    timers('backward-embedding-all-reduce').stop()

    if DEBUG_GRAD:
        string = f"=================== grad info AFTER sync [rank {torch.distributed.get_rank()}] ==================="
        with open(f"{args.log_path}{args.log_name}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
            f.write(string+"\n")    
        for name, params in model[0].named_parameters():
            string = f"[DEBUG] param name {name}, grad_requires: {params.requires_grad},\n weight: {params.data} \n grad_value: {params.main_grad}"
            with open(f"{args.log_path}{args.log_name}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
                f.write(string+"\n")   

    # Update parameters.
    timers('optimizer').start()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()    
    timers('optimizer').stop()        

    # Empty unused memory
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size                  
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad

elapsed_time_per_iteration_list = []
def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad, model=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []
    timers_to_recored = []
    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    def add_to_record(name):
        if name in timers.timers:
            timers_to_recored.append(name)
    
    add_to_logging('forward-backward')
    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-send-backward-recv')
    add_to_logging('forward-send-forward-recv')
    add_to_logging('forward-backward-send-forward-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-send-backward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')

    #================== Record timers ==================
    add_to_logging("reshape")
    add_to_logging("partition")
    add_to_logging("create_recv_placeholder")
    add_to_logging("tensor_send_next")
    add_to_logging("tensor_send_prev")
    add_to_logging("tensor_recv_next")
    add_to_logging("tensor_recv_prev")
    #================== Record timers ==================



    add_to_record('forward-compute')
    add_to_record('forward-recv')
    add_to_record('forward-send')
    add_to_record('forward-send-backward-recv')
    add_to_record('forward-send-forward-recv')
    add_to_record('forward-backward-send-forward-backward-recv')
    add_to_record('backward-compute')
    add_to_record('backward-recv')
    add_to_record('backward-send')
    add_to_record('backward-send-forward-recv')
    add_to_record('backward-send-backward-recv')
    add_to_record('backward-params-all-reduce')
    add_to_record('backward-embedding-all-reduce')
    add_to_record('optimizer')
    add_to_record('batch-generator')
    add_to_record("reshape")
    add_to_record("partition")
    add_to_record("create_recv_placeholder")
    add_to_record("tensor_send_next")
    add_to_record("tensor_send_prev")
    add_to_record("tensor_recv_next")
    add_to_record("tensor_recv_prev")



    model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module)) 

    # Calculate batch size.
    batch_size = args.micro_batch_size * get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0 ) and \
       is_last_rank():
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer and torch.distributed.get_rank() == 0:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        global elapsed_time_per_iteration_list
        elapsed_time_per_iteration_list.append(elapsed_time_per_iteration * 1000.0)
        if(len(elapsed_time_per_iteration_list) > 1):
            log_string += ' average elapsed time per iteration (ms): {:.1f} |'.format(
                sum(elapsed_time_per_iteration_list[1:]) / (len(elapsed_time_per_iteration_list)-1))
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        # print_rank_last(log_string)
        # modify to print in all rank
        if (args.rank == 0):
            print(log_string, flush=True)


        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        _time_to_csv = timers.log(timers_to_log, normalizer=args.log_interval)
        timers.save_record_time(timers_to_recored,iteration)
        if iteration == (args.train_iters ):
            time_to_csv = [["global_batch_size", "time"] + _time_to_csv[0], [batch_size, f"{elapsed_time_per_iteration * 1000.0:.2f}"] + _time_to_csv[1]]
            if(os.path.exists(f"{args.log_path}csv/") == False):
                os.makedirs(f"{args.log_path}csv/",exist_ok=True)
            with open(f"{args.log_path}csv/{args.log_name}_stage{mpu.get_pipeline_model_parallel_rank()}_rank{torch.distributed.get_rank()}.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                for row in time_to_csv:
                    writer.writerow(row)
            # nane | mbs_idx | start_time | end_time
        if iteration == args.train_iters :
            _time_to_csv_record = timers.get_record_time()
            if(args.rank == 0 and args.record_path is not None and args.record_prefix is not None):
                if(os.path.exists(args.record_path) == False):
                    os.makedirs(f"{args.record_path}",exist_ok=True)
                with open(f"{args.record_path}/{args.record_prefix}_stage{mpu.get_pipeline_model_parallel_rank()}_rank{torch.distributed.get_rank()}_record_.csv", mode="w", newline="") as file:
                    writer = csv.writer(file)
                    for row in _time_to_csv_record:
                        writer.writerow(row)
    return report_memory_flag, elapsed_time_per_iteration * 1000.0


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save-checkpoint').start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save-checkpoint').stop()
    timers.log(['save-checkpoint'])


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    memoryer = get_memory_record()
    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    
    timers('interval-time').start()
    # print_datetime('before the start of training step')
    report_memory_flag = True

    while iteration < args.train_iters:
        # print(f"iteration {iteration}")
        start_time = time.time()
        torch.cuda.synchronize()
        if(iteration >=args.system_warm):
            memoryer.warm = False
            timers.warm = False
            torch.cuda.cudart().cudaProfilerStart()

        update_num_microbatches(args.consumed_train_samples)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       lr_scheduler,iteration)
        if(iteration >=args.system_warm and args.record_path is not None and args.record_prefix is not None):
            if not os.path.exists(args.record_path):
                os.makedirs(args.record_path, exist_ok=True)
            memoryer.save(f"{args.record_path}/{args.record_prefix}_memory_{iteration}_rank_{torch.distributed.get_rank()}.log")
            # timers.save(f"{args.record_path}/{args.record_prefix}_time_{iteration}_rank_{torch.distributed.get_rank()}.time")
            torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        iteration += 1

        args.consumed_train_samples += args.micro_batch_size * \
                                    get_num_microbatches()                                            

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag, iteration_time = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad, model)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Evaluation
        # if args.eval_interval and iteration % args.eval_interval == 0 and \
        #    args.do_valid:
        #     prefix = 'iteration {}'.format(iteration)
        #     evaluate_and_print_results(prefix, forward_step_func,
        #                                valid_data_iterator, model,
        #                                iteration, False)

        # Checkpointing
        saved_checkpoint = False
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     lr_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         lr_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()
        if(torch.distributed.get_rank() == 0):
            print(f"iteration {iteration}: time = {time.time() - start_time}")

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    forward_backward_func = forward_backward_pipelining_with_interleaving
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving
            else:
                forward_backward_func = forward_backward_no_pipelining
            loss_dicts = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += args.micro_batch_size \
                                        * get_num_microbatches()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
            args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group. 
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                        args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                        eval_iters * args.global_batch_size,
                                        test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        # do_train = train_dataloader is not None and args.train_iters > 0
        # do_valid = valid_dataloader is not None and args.eval_iters > 0
        # do_test = test_dataloader is not None and args.eval_iters > 0

        do_train = args.train_iters > 0
        do_valid = args.eval_iters > 0
        do_test = args.eval_iters > 0

        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if(args.distributed_backend == "gloo"):
        flags = flags.cpu()
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()


    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
