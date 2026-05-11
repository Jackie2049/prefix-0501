# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/pretrain_gpt.py
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

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import FlexGPTModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = FlexGPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator,current_data_idx):
    """Generate a batch"""

    ## Currently using synthetic data
    args = get_args()
    vocab_size = 50257
    seq = args.mbs_seq[args.current_pipe][current_data_idx][1]// mpu.get_op_cp_size(0)
    tokens = torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0]//mpu.get_op_dp_size(0), seq), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    # labels = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0]//mpu.get_op_dp_size(-1), seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0], 1,seq, seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0]//mpu.get_op_dp_size(0),seq), requires_grad=False, device=torch.cuda.current_device()).long() * seq

    return tokens, loss_mask, position_ids, attention_mask

import json
from typing import List, Dict


def read_json_file(file_path: str) -> List[Dict]:
    import json
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_file = json.loads(line.strip())
            data.append(json_file["input"]+" "+json_file["output"])
    return data



def distribute_data(tokenized_data):
    args = get_args()
    current_pipe = args.current_pipe
    pipe_num = args.pipe_num

    distributed_data = []
    for i in range(len(tokenized_data)):
        if i % pipe_num == current_pipe:
            distributed_data.append(tokenized_data[i])
    return distributed_data


def process_data():
    args = get_args()
    data = torch.load("/workspace/heteflex/tools/token_1.pt")
    res = distribute_data(data)
    return res

current_data_idx =0    


def get_batch(data_iterator,_):
    """Generate a batch"""
    global current_data_idx
    ## Currently using synthetic data
    args = get_args()
    data = args.distributed_data
    vocab_size = 50257
    mbs =  args.micro_batch_size
    max_len = 0
    for data_ in data[current_data_idx*mbs:(current_data_idx+1)*mbs]:
        max_len = max(max_len, len(data_))
    args.mbs_seq[args.current_pipe][current_data_idx][0] = mbs
    args.mbs_seq[args.current_pipe][current_data_idx][1] = max_len
    seq = args.mbs_seq[args.current_pipe][current_data_idx][1]// mpu.get_op_cp_size(0)
    tokens = torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0], seq), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0], seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0], 1,seq, seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((args.mbs_seq[args.current_pipe][current_data_idx][0],seq), requires_grad=False, device=torch.cuda.current_device()).long() * seq
    current_data_idx+=1
    print(f"input data shape {tokens.shape} for rank {torch.distributed.get_rank()}")
    return tokens, loss_mask, position_ids, attention_mask



def get_batch(data_iterator,_):
    """Generate a batch"""
    global current_data_idx
    ## Currently using synthetic data
    args = get_args()
    data = args.distributed_data
    cache_ratio = args.cache_ratio_list
    vocab_size = 50257
    max_len = 0
    
    data_ = data[current_data_idx]
    
    for d in data_:
        max_len = max(max_len, len(d))
    padded_data = []
    for d in data_:
        padded_d = d + [0]*(max_len - len(d))
        padded_data.append(padded_d)
    
    # seq = max_len
    seq = int(max_len * (1 - cache_ratio[current_data_idx]/3 ))
    mbs = len(data_)
    # tokens = torch.tensor(padded_data, requires_grad=False, device=torch.cuda.current_device()).long()
    
    tokens = torch.rand((mbs, seq), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((mbs, seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((mbs, 1,seq, seq), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((mbs,seq), requires_grad=False, device=torch.cuda.current_device()).long() * seq
    current_data_idx+=1
    # print(f"len(data) {len(data)} input data shape {tokens.shape} for rank {torch.distributed.get_rank()}")
    return tokens, loss_mask, position_ids, attention_mask



def get_batch_packing_warm(data_iterator,_):
    """Generate a batch"""
    global current_data_idx
    ## Currently using synthetic data
    args = get_args()
    data = args.distributed_data
    vocab_size = 50257
    max_len = 0
    shared_prefix_len = args.shared_prefix_len[current_data_idx]
    store_for_sample_idx = args.store_for_sample_idx[current_data_idx]
    shared_for_sample_idx = args.shared_for_sample_idx[current_data_idx]
    data_ = data[current_data_idx]
    
    sp_size = mpu.get_op_cp_size(0)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    total_len = 0
    cu_seq_lens = [0]
    for idx,d in enumerate(data_):
        effective_len = len(d)//sp_size
        if(args.share_activation):
            effective_len -= (shared_prefix_len[idx]//sp_size)
        if(sp_size >1 and args.zigzag_ring_attn): #effective len should be multiple of 2
            effective_len = ((effective_len +1)//2)*2
        total_len += effective_len
        cu_seq_lens.append(total_len)
        max_len = max(max_len, effective_len)

    cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device())
    if(total_len % tp_size != 0):
        pad_len = tp_size - (total_len % tp_size)
        total_len += pad_len
        cu_seq_lens[-1] += pad_len
    k_cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device()).clone().detach()
    max_seqlen_k = max_len
    tokens = torch.rand((1, total_len), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((1, total_len), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((1, 1,total_len, total_len), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((1,total_len), requires_grad=False, device=torch.cuda.current_device()).long() * total_len
    # current_data_idx+=1
    # print("packing cu_seq_lens:", cu_seq_lens)
    # print("packing k_cu_seq_lens:", k_cu_seq_lens)
    # print("1max_seqlen_k:", max_seqlen_k)
    # print(f"len(data) {len(data)} input data shape {tokens.shape} for rank {torch.distributed.get_rank()}")
    return tokens, loss_mask, position_ids, attention_mask , shared_prefix_len , cu_seq_lens, k_cu_seq_lens, max_len, max_seqlen_k, None, None, None, None, current_data_idx


def get_batch_packing(data_iterator,_):
    """Generate a batch"""
    global current_data_idx
    ## Currently using synthetic data
    args = get_args()
    data = args.distributed_data
    vocab_size = 50257
    max_len = 0
    shared_prefix_len = args.shared_prefix_len[current_data_idx]
    store_for_sample_idx = args.store_for_sample_idx[current_data_idx]
    shared_for_sample_idx = args.shared_for_sample_idx[current_data_idx]
    data_ = data[current_data_idx]
    
    sp_size = mpu.get_op_cp_size(0)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    total_len = 0
    cu_seq_lens = [0]
    original_len = 0
    k_cu_seq_lens = [0]
    for idx,d in enumerate(data_):
        effective_len = len(d)//sp_size
        original_len += effective_len
        if(args.share_activation):
            effective_len -= (shared_prefix_len[idx]//sp_size)
        if(sp_size >1 and args.zigzag_ring_attn): #effective len should be multiple of 2
            effective_len = ((effective_len +1)//2)*2
        total_len += effective_len
        # if(effective_len >0):
        k_cu_seq_lens.append(original_len)
        cu_seq_lens.append(total_len)
        max_len = max(max_len, effective_len)
        # else:
        #     print("effective_len:", effective_len)
    if(total_len % tp_size != 0):
        pad_len = tp_size - (total_len % tp_size)
        total_len += pad_len
        cu_seq_lens[-1] += pad_len
        k_cu_seq_lens[-1] += pad_len
        max_len = max(max_len, pad_len)

    if(total_len ==0):
        print("len(data_):", len(data_))
        print(" sum shared_prefix_len:", sum(shared_prefix_len))
        print("sum len(data_):", sum([ len(d) for d in data_]))
        print("error: total_len =0 at data idx ", current_data_idx)
        exit(0) 
    cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device())
    k_cu_seq_lens = torch.tensor(k_cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device())
    max_seqlen_k = (k_cu_seq_lens[1:] - k_cu_seq_lens[:-1]).max().item()
    tokens = torch.rand((1, total_len), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((1, total_len), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((1, 1,total_len, total_len), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((1,total_len), requires_grad=False, device=torch.cuda.current_device()).long() * total_len
    current_data_idx+=1

    if(len(data)>current_data_idx):
        shared_prefix_len_ = args.shared_prefix_len[current_data_idx]
        data_ = data[current_data_idx]
        total_len = 0
        next_cu_seq_lens = [0]
        original_len = 0
        next_k_cu_seq_lens = [0]
        for idx,d in enumerate(data_):
            effective_len = len(d)//sp_size
            original_len += effective_len
            next_k_cu_seq_lens.append(original_len)
            if(args.share_activation):
                effective_len -= (shared_prefix_len_[idx]//sp_size)
            if(sp_size >1 and args.zigzag_ring_attn): #effective len should be multiple of 2
                effective_len = ((effective_len +1)//2)*2
            total_len += effective_len
            next_cu_seq_lens.append(total_len)
        if(total_len % tp_size != 0):
            pad_len = tp_size - (total_len % tp_size)
            total_len += pad_len
            next_cu_seq_lens[-1] += pad_len
            next_k_cu_seq_lens[-1] += pad_len
        next_cu_seq_lens = torch.tensor(next_cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device())
        next_k_cu_seq_lens = torch.tensor(next_k_cu_seq_lens, dtype=torch.int32, device=torch.cuda.current_device())
        next_max_seqlen_k = (next_k_cu_seq_lens[1:] - next_k_cu_seq_lens[:-1]).max().item()
        next_max_seqlen_q = (next_cu_seq_lens[1:] - next_cu_seq_lens[:-1]).max().item()
    else:
        next_cu_seq_lens = None
        next_k_cu_seq_lens = None
        next_max_seqlen_k = None
        next_max_seqlen_q = None
    # print("packing cu_seq_lens:", cu_seq_lens)
    # print("packing k_cu_seq_lens:", k_cu_seq_lens)
    # print("1max_seqlen_k:", max_seqlen_k)
    # print(f"len(data) {len(data)} input data shape {tokens.shape} for rank {torch.distributed.get_rank()}")
    return tokens, loss_mask, position_ids, attention_mask , shared_prefix_len , cu_seq_lens, k_cu_seq_lens, max_len, max_seqlen_k, next_cu_seq_lens, next_k_cu_seq_lens, next_max_seqlen_q, next_max_seqlen_k, current_data_idx-1

# def get_batch(data_iterator):
#     """Generate a batch"""
#     rank = torch.distributed.get_rank()
#     args = get_args()
#     micro_batch_size = args.micro_batch_size
#     if(rank == 0):
#         print("set batch = 2")
#         micro_batch_size = 4
#     else:
#         print("set batch = 2")
#         micro_batch_size = 2
#     ## Currently using synthetic data
#     vocab_size = 50257
#     tokens = torch.rand((micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
#     # labels = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
#     loss_mask =  (torch.rand((micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
#     attention_mask = (torch.rand((micro_batch_size, 1, args.seq_length, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
#     position_ids = torch.rand((micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length
#     return tokens, loss_mask, position_ids, attention_mask

def loss_func(loss_mask, output_tensor):
    losses = output_tensor["output"].float()
    loss_mask = loss_mask.view(-1).float()
    # print("losses shape:", losses.shape, "loss_mask shape:", loss_mask.shape)
    loss_mask = (torch.rand(losses.shape, requires_grad=False, device=torch.cuda.current_device())< 0.5).view(-1).float()
    # print("after check loss shape:", losses.shape, "loss_mask shape:", loss_mask.shape)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model, extra_tensors_,current_data_idx,iteration=None):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch. 
    timers('batch-generator').start()
    extra_tensors = {}
    if(args.packing):
        if(args.system_warm > iteration):
            tokens, loss_mask, position_ids, attention_mask , cache_len , cu_seq_lens, k_cu_seq_lens, max_seqlen_q, max_seqlen_k, next_cu_seq_lens, next_k_cu_seq_lens, next_max_seqlen_q, next_max_seqlen_k, current_data_idx = get_batch_packing_warm(data_iterator,current_data_idx)
            extra_tensors["system_warm"] = True
        else:
            tokens, loss_mask, position_ids, attention_mask , cache_len , cu_seq_lens, k_cu_seq_lens, max_seqlen_q, max_seqlen_k, next_cu_seq_lens, next_k_cu_seq_lens, next_max_seqlen_q, next_max_seqlen_k, current_data_idx = get_batch_packing(data_iterator,current_data_idx)
            extra_tensors["system_warm"] = False
    else:
        tokens, loss_mask, position_ids, attention_mask = get_batch(data_iterator,current_data_idx)
        cu_seq_lens = None
        cache_len = None
        k_cu_seq_lens = None
        max_seqlen_q = None
        max_seqlen_k = None
    input_tensors = {}
    input_tensors["enc_input_ids"] = tokens
    input_tensors["enc_position_ids"] = position_ids
    extra_tensors["enc_cache_len"] = cache_len
    extra_tensors["enc_cu_seq_lens"] = cu_seq_lens
    extra_tensors["enc_k_cu_seq_lens"] = k_cu_seq_lens
    extra_tensors["enc_attention_mask"] = attention_mask
    extra_tensors["max_seqlen_q"] = max_seqlen_q
    extra_tensors["max_seqlen_k"] = max_seqlen_k
    extra_tensors["enc_next_cu_seq_lens"] = next_cu_seq_lens
    extra_tensors["enc_next_k_cu_seq_lens"] = next_k_cu_seq_lens
    extra_tensors["next_max_seqlen_q"] = next_max_seqlen_q
    extra_tensors["next_max_seqlen_k"] = next_max_seqlen_k
    extra_tensors["current_data_idx"] = current_data_idx
    # extra_tensors["labels"] = labels
    if extra_tensors_ is not None:
        for key in extra_tensors_:
            extra_tensors[key] = extra_tensors_[key]
            
    timers('batch-generator').stop()

    if mpu.is_pipeline_last_stage():
        output_tensor = model(input_tensors, extra_tensors)
        ouput_extra_tensors = None
    else:
        output_tensor, ouput_extra_tensors = model(input_tensors, extra_tensors)

    return output_tensor, ouput_extra_tensors, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    ## Currently using synthetic data
    return None, None, None

    # args = get_args()
    # print_rank_0('> building train, validation, and test datasets '
    #             'for GPT ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     data_impl=args.data_impl,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     seq_length=args.seq_length,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup))
    # print_rank_0("> finished creating GPT datasets ...")

    # return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    forward_step_func = forward_step

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
