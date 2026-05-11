import torch
import os
import json
from megatron import get_memory_record
class MemoryManager():
    def __init__(self, layer_idx, tp_size, store_for_sample_idx, shared_for_sample_idx, args):
        self.layer_idx = layer_idx
        self.store_for_sample_idx = store_for_sample_idx    
        self.shared_for_sample_idx = shared_for_sample_idx
        self.args = args
        self.store_activation_dict = {}
        self.prefetch_kv = None
        self.hidden_size = args.hidden_size // args.num_attention_heads
        self.num_attention_heads = args.num_attention_heads//tp_size
        self.prefetch_cu_seq_lens = None
        self.prefetch_max_seqlen = None
        self.prefetch_stream = torch.cuda.Stream()
        self.pending_free = []
        self.gc_stream = torch.cuda.Stream()
        self.save_activation_size_MB = 0
        self.save_all_activation_size_MB = 0
        self.hit_count = 0
        self.miss_count = 0
        self.hit_rate_list = []
        self.intra_share_rate = []
        self.inter_share_rate = []
        self.memory_record  = get_memory_record()


    def store_activation(self, tensor, batch_idx, k_cu_seq_lens):
        # return
        sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][0]
        for idx in range(len(self.store_for_sample_idx[batch_idx])):
            dict_ = self.store_for_sample_idx[batch_idx][idx]
            if(len(dict_)==0):
                continue
            # print("Storing activation for layer:" " sample_idx:", sample_idx+idx, " dict_:", dict_)
            base_address = dict_["base_address"]
            end_address = dict_["end_address"]
            last_sample_idx = dict_["last_sample_idx"]
            sample_start_address = k_cu_seq_lens[idx]
            save_tensor = tensor[sample_start_address+base_address:sample_start_address+end_address,:,:,:].clone().detach()
            self.store_activation_dict[sample_idx+idx] = { "tensor": save_tensor,
                                                            "owner_sample_idx": sample_idx+idx,
                                                            "last_sample_idx": last_sample_idx,
                                                            "base_address": base_address,
                                                            }
            self.save_activation_size_MB += save_tensor.nelement() * tensor.element_size() / 1024**2
        self.save_all_activation_size_MB += tensor.nelement() * tensor.element_size() / 1024**2
        if(self.args.rank == 0 and self.layer_idx == 0 and self.args.record_path is not None and self.args.record_prefix is not None):
            self.memory_record.record_activation("forward","saved_activation_size_MB", self.save_activation_size_MB)
            self.memory_record.record_activation("forward","all_saved_activation_size_MB", self.save_all_activation_size_MB)
            self.memory_record.record_activation("forward","hit_rate", self.hit_rate_list[-1])

    def retrieve_activation_ideal(self, tensor, batch_idx, q_cu_seq_lens, k_cu_seq_lens):
        total_seq_len =  k_cu_seq_lens[-1]
        result_tensors = torch.empty((total_seq_len, 2, self.num_attention_heads, self.hidden_size), device=tensor.device, dtype=tensor.dtype)

        return result_tensors

    def retrieve_activation(self, tensor, batch_idx, q_cu_seq_lens, k_cu_seq_lens):

        if(self.args.save_all_activation):
            total_seq_len =  k_cu_seq_lens[-1]
            return torch.empty((total_seq_len, 2, self.num_attention_heads, self.hidden_size), device=tensor.device, dtype=tensor.dtype)
            

        sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][0]
        end_sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][1]
        with torch.cuda.stream(self.prefetch_stream):
            total_seq_len =  k_cu_seq_lens[-1]
            result_tensors = torch.empty((total_seq_len, 2, self.num_attention_heads, self.hidden_size), device=tensor.device, dtype=tensor.dtype)
            add_seq_len = 0
            for idx, dict_ in enumerate(self.shared_for_sample_idx[batch_idx]):
                old_start = q_cu_seq_lens[idx]
                old_end = q_cu_seq_lens[idx+1]
                new_start = k_cu_seq_lens[idx]
                current_sample_idx = sample_idx + idx
                result_tensors[new_start:new_start+(old_end - old_start),:,:,:] = tensor[old_start:old_end ,:,:,:]
                for share_request_idx in dict_:
                    start = dict_[share_request_idx][0]
                    end = dict_[share_request_idx][1]
                    if(share_request_idx >= sample_idx):
                        share_request_idx_local_idx = share_request_idx - sample_idx
                        start = dict_[share_request_idx][0]
                        end = dict_[share_request_idx][1]
                        shared_part_start = k_cu_seq_lens[share_request_idx_local_idx]
                        result_tensors[new_start+start:new_start+end,:,:,:] = result_tensors[shared_part_start+start:shared_part_start+end,:,:,:]
                        continue
                    store_activation = self.store_activation_dict[share_request_idx]
                    share_request_idx_base_address = store_activation["base_address"]
                    # 在 tensor 的 cu_seq_lens[idx+1] 位置增加对应的长度 位置加上（end-start）的长度
                    # result_tensors[old_start+add_seq_len:old_start+add_seq_len+(end - start)] = 1
                    result_tensors[old_start+add_seq_len:old_start+add_seq_len+(end - start)] = store_activation["tensor"][start - share_request_idx_base_address:end - share_request_idx_base_address]
                    add_seq_len += (end - start)
        return result_tensors

    def get_original_length(self, batch_idx, cu_seq_lens):
        cu_seq_lens_ = cu_seq_lens.clone()
        add_seq_len = 0
        for idx, dict_ in enumerate(self.shared_for_sample_idx[batch_idx]):
            for key in dict_:
                start = dict_[key][0]
                end = dict_[key][1]
                add_seq_len += (end - start)
            cu_seq_lens_[idx+1] += add_seq_len
        max_seqlen = (cu_seq_lens_[1:] - cu_seq_lens_[:-1]).max().item()
        total_seq_len = cu_seq_lens_[-1]
        return cu_seq_lens_,  max_seqlen , total_seq_len
    
    def retrieve_activation_prefetch(self, batch_idx, q_cu_seq_lens,k_cu_seq_lens,device,dtype):

        if(batch_idx+1)>=len(self.shared_for_sample_idx):
            self.prefetch_kv = None
            return
        sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][0]
        end_sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][1]
        with torch.cuda.stream(self.prefetch_stream):
        # print("input cu_seq_lens:", cu_seq_lens)
            total_seq_len =  k_cu_seq_lens[-1]
            result_tensors = torch.empty((total_seq_len, 2, self.num_attention_heads, self.hidden_size), device=device, dtype=dtype)
            add_seq_len = 0
            for idx, dict_ in enumerate(self.shared_for_sample_idx[batch_idx]):
                old_start = q_cu_seq_lens[idx]
                old_end = q_cu_seq_lens[idx+1]
                current_sample_idx = sample_idx + idx
                for share_request_idx in dict_:
                    start = dict_[share_request_idx][0]
                    end = dict_[share_request_idx][1]
                    if(share_request_idx >= sample_idx):
                        # print("Error: share_request_idx >= sample_idx in prefetching")
                        continue
                    store_activation = self.store_activation_dict[share_request_idx]
                    share_request_idx_base_address = store_activation["base_address"]
                    # 在 tensor 的 cu_seq_lens[idx+1] 位置增加对应的长度 位置加上（end-start）的长度
                    # result_tensors[old_start+add_seq_len:old_start+add_seq_len+(end - start)] = 1
                    # result_tensors[old_start+add_seq_len:old_start+add_seq_len+(end - start)] = store_activation["tensor"][start - share_request_idx_base_address:end - share_request_idx_base_address]
                    add_seq_len += (end - start)
            self.prefetch_kv = result_tensors


    def get_from_prefetch(self, tensor, batch_idx, q_cu_seq_lens, k_cu_seq_lens):

        if(q_cu_seq_lens is not None):
            self.hit_rate_list.append(1-q_cu_seq_lens[-1].item()/k_cu_seq_lens[-1].item())
        
        if(self.prefetch_kv is None):
            return self.retrieve_activation(tensor, batch_idx, q_cu_seq_lens, k_cu_seq_lens)
        sample_idx = self.args.batch_idx_mapping_sample_idx[batch_idx][0]
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        for idx, dict_ in enumerate(self.shared_for_sample_idx[batch_idx]):
            # old_start = q_cu_seq_lens[idx]
            # old_end = q_cu_seq_lens[idx+1]
            new_start = k_cu_seq_lens[idx]
            # self.prefetch_kv[new_start:new_start+(old_end - old_start),:,:,:] = tensor[old_start:old_end ,:,:,:]
            for share_request_idx in dict_.keys():
                if(share_request_idx >= sample_idx):
                    share_request_idx_local_idx = share_request_idx - sample_idx
                    start = dict_[share_request_idx][0]
                    end = dict_[share_request_idx][1]
                    shared_part_start = k_cu_seq_lens[share_request_idx_local_idx]
                    self.prefetch_kv[new_start+start:new_start+end,:,:,:] = self.prefetch_kv[shared_part_start+start:shared_part_start+end,:,:,:]
        return self.prefetch_kv

    def clear_activation(self, current_batch_idx):
        if(self.args.save_all_activation):
            return
        to_delete = []
        current_sample_idx = self.args.batch_idx_mapping_sample_idx[current_batch_idx][1]
        for sample_idx, entry in self.store_activation_dict.items():
            last_sample_idx = entry["last_sample_idx"]

            # 1. 语义判断：未来不会再被用
            if last_sample_idx < current_sample_idx:
                tensor = entry["tensor"]

                # 2. 记录一个 CUDA event，确保之前的使用完成
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())

                # 3. 放入 pending_free 队列
                self.pending_free.append((event, tensor, sample_idx))
                to_delete.append(sample_idx)

                self.save_activation_size_MB -= tensor.nelement() * tensor.element_size() / 1024**2
        # 4. 从 dict 中移除（不等于 free）
        for sample_idx in to_delete:
            del self.store_activation_dict[sample_idx]

    def async_gc(self):
        """
        Try to free activations whose CUDA events are completed.
        This runs asynchronously and never blocks compute.
        """

        if not self.pending_free:
            return

        remaining = []

        with torch.cuda.stream(self.gc_stream):
            for event, tensor, sample_idx in self.pending_free:
                # event 已完成，说明所有使用这个 tensor 的 kernel 都结束了
                if event.query():
                    # 真正释放：删掉 Python reference
                    del tensor
                else:
                    remaining.append((event, tensor, sample_idx))

        self.pending_free = remaining