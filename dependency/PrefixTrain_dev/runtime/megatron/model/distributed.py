# coding=utf-8
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

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.utils import unwrap_model
from .module import Float16Module

import os
LOG_NAME = os.environ.get("LOG_NAME", None)

class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel_Profile(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel_Profile, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.     

            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)
        
        args = get_args()
        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.resharding = args.resharding_stages[rank_in_pipeline]

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

    ## TODO: continious buffer with resharding.
    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        # args = get_args()
        if self._grad_buffers is not None:
            if self.resharding:
                raise RuntimeError("cross-op resharding with continues buffer is not supported yet.")
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=mpu.get_data_parallel_group())
        else:
            if self.resharding:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                dp_groups = {}
                dp_sizes = {}
                # Pack the buckets.
                model_ = unwrap_model(self.module, (Float16Module)) 
                for op in model_.language_model.ops:
                    tp_size = op.tp_size
                    dp_size = op.dp_size
                    for param in op.parameters():
                        if param.requires_grad and param.grad is not None:
                            data_type = param.data.type()
                            key_str = str(data_type)+str(tp_size)+str(dp_size)
                            if key_str not in buckets:
                                buckets[key_str] = []
                            buckets[key_str].append(param)
                            param.main_grad = param.grad

                            if key_str not in dp_groups:
                                dp_groups[key_str] = mpu.get_data_parallel_group_via_op_index(op.op_index)
                                dp_sizes[key_str] = dp_size

                # For each bucket, all-reduce and copy all-reduced grads.
                for key_str in buckets:
                    bucket = buckets[key_str]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= dp_sizes[key_str]
                    torch.distributed.all_reduce(
                        coalesced, group=dp_groups[key_str])
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)
            else:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                # Pack the buckets.
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.type()
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                        param.main_grad = param.grad

                # For each bucket, all-reduce and copy all-reduced grads.
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)                

def write_to_txt(rank, string):
    with open(f"{LOG_NAME}_rank_{rank}.log", "a+") as f:
        f.write(string+"\n")

class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)
        rank = torch.distributed.get_rank()
        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers
        self.rank = torch.distributed.get_rank()
        self.start_op_index = mpu.get_current_ops_start_index()
        self.all_reduce_rank_each_op = mpu.get_global_data_parallel_each_op()
        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffers_op = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffers_op = []
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            type_num_elements_op={}
            # print(f"rank: {torch.distributed.get_rank()}, len(self.module.parameters) {len(list(self.module.parameters()))}")
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()
            op_idx = 0
            for op in self.module.module.language_model.ops:
                op_param_size = 0
                for param in op.parameters():
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        op_param_size += param.data.nelement()
                type_num_elements_op[op_idx] = op_param_size
                op_idx += 1

            # Allocate the buffer.
            # for dtype, num_elements in type_num_elements.items():
            #     self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            self.all_reduce_tensor_size = torch.full([len(self.all_reduce_rank_each_op)], torch.iinfo(torch.int32).max , dtype=torch.int32)
            op_idx = 0
            for op in self.module.module.language_model.ops:
                op_param_size = 0
                for param in op.parameters():
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        op_param_size += param.data.nelement()
                self._grad_buffers_op.append(MemoryBuffer(op_param_size, dtype))
                # if(rank == 0 or rank == 4):
                #     print(f"rank: {rank}, op_idx: {op_idx}, op_param_size: {op_param_size}")
                self.all_reduce_tensor_size[self.start_op_index+op_idx] = op_param_size/(len(self.all_reduce_rank_each_op[self.start_op_index+op_idx][rank]))
                op_idx += 1


            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.     
            self.all_reduce_tensor_size = self.all_reduce_tensor_size.cuda()
            torch.distributed.all_reduce(self.all_reduce_tensor_size,op=torch.distributed.ReduceOp.MIN)
            op_idx = 0
            for op in self.module.module.language_model.ops:
                for param in op.parameters():
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        type_num_elements_op[op_idx] -= param.data.nelement()
                        param.main_grad = self._grad_buffers_op[op_idx].get(
                            param.data.shape, type_num_elements_op[op_idx])
    
                op_idx += 1


            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.

            for op in self.module.module.language_model.ops:
                for param in op.parameters():
                    if param.requires_grad:
                        param_tmp = param.expand_as(param)
                        # Get the gradient accumulator functtion.
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        grad_acc.register_hook(self._make_param_hook(param))
                        self.grad_accs.append(grad_acc)
                        
        args = get_args()
        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.resharding = args.resharding_stages[rank_in_pipeline]
        torch.cuda.synchronize()
        # if(rank ==0 or rank == 4):
            # print(f"self.all_reduce_tensor_size: {self.all_reduce_tensor_size}")
    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        # assert self._grad_buffers is not None, 'buffers are not initialized.'
        # for _, buffer_ in self._grad_buffers.items():
        #     buffer_.zero()
        for buffer_ in self._grad_buffers_op:
            buffer_.zero()


    ## TODO: continious buffer with resharding.
    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        # args = get_args()
        if self._grad_buffers_op is not None:
            if self.resharding:
                raise RuntimeError("cross-op resharding with continues buffer is not supported yet.")
            # print("len(self._grad_buffers): ", len(self._grad_buffers))\

            # print(f"rank: {rank}, start_op_index {start_op_index} len(self._grad_buffers_op): {len(self._grad_buffers_op)}")
            
            for i in range(len(self.all_reduce_rank_each_op)): #第i个op
                group_rank_in_op  = self.all_reduce_rank_each_op[i]
                # print(group_rank_in_op)
                if(self.rank not in group_rank_in_op or  i == 0):
                    continue
                group_rank = group_rank_in_op[self.rank]
                buffer_shard_list = []
                buffer_ = self._grad_buffers_op[i-self.start_op_index]
                num_all_reduce = len(group_rank)
                for j in range(len(group_rank)): #第j个group
                    shape = buffer_.data.shape
                    if(shape[0] == 0):
                        continue
                    modify_shape =  torch.Size([self.all_reduce_tensor_size[i]])
                    buffer = buffer_.get(modify_shape,0)
                    group = torch.distributed.new_group(group_rank[j])
                    write_to_txt(self.rank, f"op {i} num_all_reduce {num_all_reduce} buffer.shape: {buffer.shape}, group_rank[j]: {group_rank[j]}")
                    torch.distributed.all_reduce(
                        buffer, group=group)

            return 
            for buffer_ in self._grad_buffers_op:
                dict_all_reduce_rank = all_reduce_rank_each_op[start_op_index]
                all_reduce_rank = dict_all_reduce_rank[rank]
                num_all_reduce = len(all_reduce_rank)
                buffer_shard_list = []
                for i in range(num_all_reduce):
                    shape = buffer_.data.shape
                    # print(f"rank: {rank}, shape: {shape}")
                    # print(f"rank: {rank}, shape.numel(): {shape.numel()}")
                    # 将shape中的第一个元素除以num_all_reduce
                    modify_shape =  torch.Size([shape[0]//num_all_reduce] )
                    if(modify_shape[0] == 0):
                        continue
                    buffer = buffer_.get(modify_shape, (num_all_reduce-i-1)*buffer_.data.numel()//num_all_reduce)
                # for i in range(num_all_reduce):   
                # print(f"rank: {torch.distributed.get_rank()}, buffer_.data.shape: {buffer_.data.shape}")
                    # buffer_.data /= mpu.get_data_parallel_world_size()
                    # torch.distributed.all_reduce(
                    #     buffer_.data, group=mpu.get_data_parallel_group())
                    # buffer_shard_list[i] /= mpu.get_data_parallel_world_size()
                    # print(f"rank: {rank}, num_all_reduce {num_all_reduce}, buffer_.data.shape: {buffer_.data.shape}, i*buffer_.data.numel()//num_all_reduce: {(num_all_reduce-i-1)*buffer_.data.numel()//num_all_reduce}")
                    data_group = torch.distributed.new_group(ranks=all_reduce_rank ,use_local_synchronization=True)
                    # torch.distributed.all_reduce(
                    #     torch.randn(1024).npu(), group=data_group)
                    
                start_op_index+=1
        else:
            print(f"self.resharding: {self.resharding}")

            if self.resharding:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                dp_groups = {}
                dp_sizes = {}
                # Pack the buckets.
                model_ = unwrap_model(self.module, (Float16Module)) 
                for op in model_.language_model.ops:
                    tp_size = op.tp_size
                    dp_size = op.dp_size
                    for param in op.parameters():
                        if param.requires_grad and param.grad is not None:
                            data_type = param.data.type()
                            key_str = str(data_type)+str(tp_size)+str(dp_size)
                            if key_str not in buckets:
                                buckets[key_str] = []
                            buckets[key_str].append(param)
                            param.main_grad = param.grad

                            if key_str not in dp_groups:
                                dp_groups[key_str] = mpu.get_data_parallel_group_via_op_index(op.op_index)
                                dp_sizes[key_str] = dp_size


                # For each bucket, all-reduce and copy all-reduced grads.
                for key_str in buckets:
                    bucket = buckets[key_str]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= dp_sizes[key_str]
                    torch.distributed.all_reduce(
                        coalesced, group=dp_groups[key_str])
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)
            else:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                # Pack the buckets.
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.type()
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                        param.main_grad = param.grad

                # For each bucket, all-reduce and copy all-reduced grads.
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)                


    # def allreduce_gradients(self):
    #     """Reduce gradients across data parallel ranks."""
    #     # If we have buffers, simply reduce the data in the buffer.
    #     if self._grad_buffers is not None:
    #         for _, buffer_ in self._grad_buffers.items():
    #             buffer_.data /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 buffer_.data, group=mpu.get_data_parallel_group())
    #     else:
    #         # Otherwise, bucketize and all-reduce
    #         buckets = {}
    #         # Pack the buckets.
    #         for param in self.module.parameters():
    #             if param.requires_grad and param.grad is not None:
    #                 tp = param.data.type()
    #                 if tp not in buckets:
    #                     buckets[tp] = []
    #                 buckets[tp].append(param)
    #                 param.main_grad = param.grad

    #         # print(f"[DEBUG] ======> allreduce_gradients <=====")
    #         # for name, params in self.module.named_parameters():
    #         #     if params.requires_grad:
    #         #         if params.grad is not None:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n main_grad: {params.main_grad}"
    #         #         else:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n grad = None"
    #         #     else:
    #         #         string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad}"
    #         #     with open(f"{LOG_NAME}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
    #         #         f.write(string+"\n")  


    #         # For each bucket, all-reduce and copy all-reduced grads.
    #         for tp in buckets:
    #             bucket = buckets[tp]
    #             grads = [param.grad.data for param in bucket]
    #             coalesced = _flatten_dense_tensors(grads)
    #             coalesced /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 coalesced, group=mpu.get_data_parallel_group())
    #             for buf, synced in zip(grads, _unflatten_dense_tensors(
    #                     coalesced, grads)):
    #                 buf.copy_(synced)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)
        self.args = get_args()
        rank = torch.distributed.get_rank()
        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers
        self.rank = torch.distributed.get_rank()
        self.start_op_index = mpu.get_current_ops_start_index()
        self.all_reduce_rank_each_op = mpu.get_global_data_parallel_each_op()
        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffers_layer = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffers_layer = {}
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            type_num_elements_layer={}
            # print(f"rank: {torch.distributed.get_rank()}, len(self.module.parameters) {len(list(self.module.parameters()))}")
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()
            # op_idx = 0
            # for op in self.module.module.language_model.ops:
            #     op_param_size = 0
            #     for param in op.parameters():
            #         if param.requires_grad:
            #             dtype = _get_buffer_type(param)
            #             op_param_size += param.data.nelement()
            #     type_num_elements_op[op_idx] = op_param_size
            #     op_idx += 1
            self.grad_accs = []

            self.start_layer_idx = self.start_op_index //self.args.num_ops_each_layer  
            self.start_layer_idx += self.start_op_index%self.args.num_ops_each_layer  !=0
            self.end_layer_idx = self.start_layer_idx+ len(self.module.module.language_model.ops)//self.args.num_ops_each_layer  -1 + (self.start_layer_idx ==0)
            self.bool_start_layer = self.start_layer_idx==0
            self.bool_end_layer = 0 

            self.all_reduce_tensor_size_layer = torch.full([len(self.all_reduce_rank_each_op)//self.args.num_ops_each_layer +2], torch.iinfo(torch.int32).max , dtype=torch.int32)

            if(self.start_layer_idx == 0):
                self.bool_end_layer =len(self.module.module.language_model.ops)%self.args.num_ops_each_layer ==3
                layer_param_size = 0
                for param in self.module.module.language_model.ops[0].parameters():
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        layer_param_size += param.data.nelement()
                type_num_elements_layer[0] = layer_param_size
                self._grad_buffers_layer[0]= MemoryBuffer(layer_param_size, dtype)
                self.all_reduce_tensor_size_layer[0] = layer_param_size/(len(self.all_reduce_rank_each_op[0][rank]))

                for param in self.module.module.language_model.ops[0].parameters():
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        param.main_grad = self._grad_buffers_layer[0].get(param.data.shape, 0)

                        param_tmp = param.expand_as(param)
                        # Get the gradient accumulator functtion.
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        grad_acc.register_hook(self._make_param_hook(param))
                        self.grad_accs.append(grad_acc)
            else:
                self.bool_end_layer =len(self.module.module.language_model.ops)%self.args.num_ops_each_layer ==2
            self.end_layer_idx += self.bool_end_layer
            print(f"rank" ,self.rank , "self.start_op_index" ,self.start_op_index, "start_layer_idx",self.start_layer_idx,"end_layer_idx",self.end_layer_idx,"len(ops)",len(self.module.module.language_model.ops))
            for i in range (max(1,self.start_layer_idx),self.end_layer_idx+1 -self.bool_end_layer):  
                layer_param_size = 0
                for op in self.module.module.language_model.ops[self.bool_start_layer+(i-max(1,self.start_layer_idx))*self.args.num_ops_each_layer :self.bool_start_layer+(i-max(1,self.start_layer_idx)+1)*self.args.num_ops_each_layer ]:
                    for param in op.parameters():
                        if param.requires_grad:
                            dtype = _get_buffer_type(param)
                            layer_param_size += param.data.nelement()
                type_num_elements_layer[i] = layer_param_size
                self._grad_buffers_layer[i] = (MemoryBuffer(layer_param_size, dtype))
                self.all_reduce_tensor_size_layer[i] = layer_param_size/(len(self.all_reduce_rank_each_op[1+(i-1)*self.args.num_ops_each_layer ][rank]))

            for i in range (max(1,self.start_layer_idx),self.end_layer_idx+1 -self.bool_end_layer):
                layer_param_size = type_num_elements_layer[i]
                for op in self.module.module.language_model.ops[self.bool_start_layer+(i-max(1,self.start_layer_idx))*self.args.num_ops_each_layer :self.bool_start_layer+(i-max(1,self.start_layer_idx)+1)*self.args.num_ops_each_layer ]:
                    for param in op.parameters():
                        if param.requires_grad:
                            dtype = _get_buffer_type(param)
                            layer_param_size -= param.data.nelement()
                            param.main_grad = self._grad_buffers_layer[i].get(param.data.shape, layer_param_size)

                            param_tmp = param.expand_as(param)
                            # Get the gradient accumulator functtion.
                            grad_acc = param_tmp.grad_fn.next_functions[0][0]
                            grad_acc.register_hook(self._make_param_hook(param))
                            self.grad_accs.append(grad_acc)

            if(self.bool_end_layer):
                layer_param_size = 0
                for op in self.module.module.language_model.ops[-2:]:
                    for param in op.parameters():
                        if param.requires_grad:
                            dtype = _get_buffer_type(param)
                            layer_param_size += param.data.nelement()
                type_num_elements_layer[self.end_layer_idx] = layer_param_size
                self._grad_buffers_layer[self.end_layer_idx]=MemoryBuffer(layer_param_size, dtype)
                self.all_reduce_tensor_size_layer[self.end_layer_idx] = layer_param_size/(len(self.all_reduce_rank_each_op[-2][rank]))

                layer_param_size = type_num_elements_layer[self.end_layer_idx]

                for op in self.module.module.language_model.ops[-2:]:
                    for param in op.parameters():
                        if param.requires_grad:
                            dtype = _get_buffer_type(param)
                            layer_param_size -= param.data.nelement()
                            param.main_grad = self._grad_buffers_layer[self.end_layer_idx].get(param.data.shape, layer_param_size)

                            param_tmp = param.expand_as(param)
                            # Get the gradient accumulator functtion.
                            grad_acc = param_tmp.grad_fn.next_functions[0][0]
                            grad_acc.register_hook(self._make_param_hook(param))
                            self.grad_accs.append(grad_acc)
            if(self.args.distributed_backend == "gloo"):
                torch.distributed.all_reduce(self.all_reduce_tensor_size_layer,op=torch.distributed.ReduceOp.MIN)
            else:
                print(f"rank {rank} start all reduce")
                self.all_reduce_tensor_size_layer = self.all_reduce_tensor_size_layer.cuda()
                torch.distributed.all_reduce(self.all_reduce_tensor_size_layer,op=torch.distributed.ReduceOp.MIN)

            self.all_reduce_group = {}
            op_idx = 0
            for layer_idx in range(0,len(self.all_reduce_rank_each_op)//self.args.num_ops_each_layer +2):
                self.all_reduce_group[layer_idx] = {}
                group_rank_in_op  = self.all_reduce_rank_each_op[op_idx]
                if(layer_idx==0):
                    op_idx+=1
                else:
                    op_idx+=self.args.num_ops_each_layer 
                # print(group_rank_in_op)
                if(layer_idx == 0):
                    continue
                if(rank not in group_rank_in_op):
                    group_rank = [[rank]]
                else:
                    group_rank = group_rank_in_op[rank]
                # print(f"rank {rank} layer_idx {layer_idx}", group_rank)
                for j in range(len(group_rank)):
                    # print(f"rank {rank} layer_idx {layer_idx} group {j} group_rank {group_rank[j]} creating")
                    self.all_reduce_group[layer_idx][j] = torch.distributed.new_group(group_rank[j])
                    # print(f"rank {rank} layer_idx {layer_idx} group {j} group_rank {group_rank[j]} created")
            # print(f"Rank {rank} finished DDP init")

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.     
            # self.all_reduce_tensor_size = self.all_reduce_tensor_size.npu()
            # torch.distributed.all_reduce(self.all_reduce_tensor_size,op=torch.distributed.ReduceOp.MIN)
            # op_idx = 0
            # for op in self.module.module.language_model.ops:
            #     for param in op.parameters():
            #         if param.requires_grad:
            #             dtype = _get_buffer_type(param)
            #             type_num_elements_op[op_idx] -= param.data.nelement()
            #             param.main_grad = self._grad_buffers_op[op_idx].get(
            #                 param.data.shape, type_num_elements_op[op_idx])
    
            #     op_idx += 1


            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            # Loop over all the parameters in the model.

            # for op in self.module.module.language_model.ops:
            #     for para in op.parameters():
            #         if para.requires_grad:
            #             para_tmp = para.expand_as(para)
            #             grad_acc = para_tmp.grad_fn.next_functions[0][0]
            #             grad_acc.register_hook(self._make_param_hook(para))
            #             self.grad_accs.append(grad_acc)
                        
        args = get_args()
        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.resharding = args.resharding_stages[rank_in_pipeline]
        torch.cuda.synchronize()
        # if(rank ==0 or rank == 4):
            # print(f"self.all_reduce_tensor_size: {self.all_reduce_tensor_size}")
        self.warm = 0
        self.iteration = 0
    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers_layer is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers_layer.items():
            buffer_.zero()



    ## TODO: continious buffer with resharding.
    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        # args = get_args()
        if self._grad_buffers_layer is not None:
            if self.resharding:
                raise RuntimeError("cross-op resharding with continues buffer is not supported yet.")
            # print("len(self._grad_buffers): ", len(self._grad_buffers))\

            # print(f"rank: {rank}, start_op_index {start_op_index} len(self._grad_buffers_op): {len(self._grad_buffers_op)}")
            
            op_idx = 0
            for layer_idx in range(0,len(self.all_reduce_rank_each_op)//self.args.num_ops_each_layer +2):

                group_rank_in_op  = self.all_reduce_rank_each_op[op_idx]
                if(layer_idx==0):
                    op_idx+=1
                else:
                    op_idx+=self.args.num_ops_each_layer 
                # print(group_rank_in_op)
                if(self.rank not in group_rank_in_op or  layer_idx == 0):
                    continue
                group_rank = group_rank_in_op[self.rank]
                buffer_ = self._grad_buffers_layer[layer_idx]
                num_all_reduce = len(group_rank)
                for j in range(len(group_rank)):
                    shape = buffer_.data.shape
                    if(shape[0]==0):
                        continue

                    modify_shape = torch.Size([self.all_reduce_tensor_size_layer[layer_idx].item()])

                    if(self.iteration>=self.warm):
                        buffer = buffer_.get(modify_shape,0) #TODO
                        # group = torch.distributed.new_group(group_rank[j],use_local_synchronization=True)
                        # write_to_txt(self.rank, f"layer {layer_idx} num_all_reduce {num_all_reduce} buffer.shape: {buffer.shape}, group_rank[j]: {group_rank[j]}")
                        if(self.args.distributed_backend == "gloo"):
                            buffer = buffer.cpu()
                        # print(f"rank {self.rank} layer_idx {layer_idx} group {j} in group {self.all_reduce_group[layer_idx][j]} all reducing")
                        torch.distributed.all_reduce(
                            buffer, group=self.all_reduce_group[layer_idx][j])
            self.iteration+=1

            # for i in range(len(self.all_reduce_rank_each_op)): #第i个op
            #     group_rank_in_op  = self.all_reduce_rank_each_op[i]
            #     # print(group_rank_in_op)
            #     if(self.rank not in group_rank_in_op or  i == 0):
            #         continue
            #     group_rank = group_rank_in_op[self.rank]
            #     buffer_shard_list = []
            #     buffer_ = self._grad_buffers_op[i-self.start_op_index]
            #     num_all_reduce = len(group_rank)
            #     for j in range(len(group_rank)): #第j个group
            #         shape = buffer_.data.shape
            #         if(shape[0] == 0):
            #             continue
            #         modify_shape =  torch.Size([self.all_reduce_tensor_size[i]])
            #         buffer = buffer_.get(modify_shape,0)
            #         group = torch.distributed.new_group(group_rank[j],use_local_synchronization=True)
            #         write_to_txt(self.rank, f"op {i} num_all_reduce {num_all_reduce} buffer.shape: {buffer.shape}, group_rank[j]: {group_rank[j]}")
            #         torch.distributed.all_reduce(
            #             buffer, group=group)

            return 
            for buffer_ in self._grad_buffers_op:
                dict_all_reduce_rank = all_reduce_rank_each_op[start_op_index]
                all_reduce_rank = dict_all_reduce_rank[rank]
                num_all_reduce = len(all_reduce_rank)
                buffer_shard_list = []
                for i in range(num_all_reduce):
                    shape = buffer_.data.shape
                    # print(f"rank: {rank}, shape: {shape}")
                    # print(f"rank: {rank}, shape.numel(): {shape.numel()}")
                    # 将shape中的第一个元素除以num_all_reduce
                    modify_shape =  torch.Size([shape[0]//num_all_reduce] )
                    if(modify_shape[0] == 0):
                        continue
                    buffer = buffer_.get(modify_shape, (num_all_reduce-i-1)*buffer_.data.numel()//num_all_reduce)
                # for i in range(num_all_reduce):   
                # print(f"rank: {torch.distributed.get_rank()}, buffer_.data.shape: {buffer_.data.shape}")
                    # buffer_.data /= mpu.get_data_parallel_world_size()
                    # torch.distributed.all_reduce(
                    #     buffer_.data, group=mpu.get_data_parallel_group())
                    # buffer_shard_list[i] /= mpu.get_data_parallel_world_size()
                    # print(f"rank: {rank}, num_all_reduce {num_all_reduce}, buffer_.data.shape: {buffer_.data.shape}, i*buffer_.data.numel()//num_all_reduce: {(num_all_reduce-i-1)*buffer_.data.numel()//num_all_reduce}")
                    data_group = torch.distributed.new_group(ranks=all_reduce_rank ,use_local_synchronization=True)
                    # torch.distributed.all_reduce(
                    #     torch.randn(1024).npu(), group=data_group)
                    
                start_op_index+=1
        else:
            print(f"self.resharding: {self.resharding}")

            if self.resharding:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                dp_groups = {}
                dp_sizes = {}
                # Pack the buckets.
                model_ = unwrap_model(self.module, (Float16Module)) 
                for op in model_.language_model.ops:
                    tp_size = op.tp_size
                    dp_size = op.dp_size
                    for param in op.parameters():
                        if param.requires_grad and param.grad is not None:
                            data_type = param.data.type()
                            key_str = str(data_type)+str(tp_size)+str(dp_size)
                            if key_str not in buckets:
                                buckets[key_str] = []
                            buckets[key_str].append(param)
                            param.main_grad = param.grad

                            if key_str not in dp_groups:
                                dp_groups[key_str] = mpu.get_data_parallel_group_via_op_index(op.op_index)
                                dp_sizes[key_str] = dp_size


                # For each bucket, all-reduce and copy all-reduced grads.
                for key_str in buckets:
                    bucket = buckets[key_str]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= dp_sizes[key_str]
                    torch.distributed.all_reduce(
                        coalesced, group=dp_groups[key_str])
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)
            else:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                # Pack the buckets.
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.type()
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                        param.main_grad = param.grad

                # For each bucket, all-reduce and copy all-reduced grads.
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)                


    # def allreduce_gradients(self):
    #     """Reduce gradients across data parallel ranks."""
    #     # If we have buffers, simply reduce the data in the buffer.
    #     if self._grad_buffers is not None:
    #         for _, buffer_ in self._grad_buffers.items():
    #             buffer_.data /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 buffer_.data, group=mpu.get_data_parallel_group())
    #     else:
    #         # Otherwise, bucketize and all-reduce
    #         buckets = {}
    #         # Pack the buckets.
    #         for param in self.module.parameters():
    #             if param.requires_grad and param.grad is not None:
    #                 tp = param.data.type()
    #                 if tp not in buckets:
    #                     buckets[tp] = []
    #                 buckets[tp].append(param)
    #                 param.main_grad = param.grad

    #         # print(f"[DEBUG] ======> allreduce_gradients <=====")
    #         # for name, params in self.module.named_parameters():
    #         #     if params.requires_grad:
    #         #         if params.grad is not None:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n main_grad: {params.main_grad}"
    #         #         else:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n grad = None"
    #         #     else:
    #         #         string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad}"
    #         #     with open(f"{LOG_NAME}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
    #         #         f.write(string+"\n")  


    #         # For each bucket, all-reduce and copy all-reduced grads.
    #         for tp in buckets:
    #             bucket = buckets[tp]
    #             grads = [param.grad.data for param in bucket]
    #             coalesced = _flatten_dense_tensors(grads)
    #             coalesced /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 coalesced, group=mpu.get_data_parallel_group())
    #             for buf, synced in zip(grads, _unflatten_dense_tensors(
    #                     coalesced, grads)):
    #                 buf.copy_(synced)
