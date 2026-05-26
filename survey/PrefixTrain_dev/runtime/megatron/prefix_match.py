from typing import List, Dict, Optional
from collections import Counter
from transformers import AutoTokenizer
import torch

# 读取jsonl
def read_json_file(file_path: str) -> List[Dict]:
    import json
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_file = json.loads(line.strip())
            data.append(json_file["input"]+" "+json_file["output"])
    return data

#  加载tokenizer 将字符串转为token列表
def load_tokenizer(tokenizer_name: str, data):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenized_data = [tokenizer.encode_plus(text, add_special_tokens=False)['input_ids'] for text in data]
    return tokenized_data

# 对每个string去除 包含所有<think> </think>及其之间的内容
def remove_think_tags(s: str) -> str:
    start_tag = "<think>"
    end_tag = "</think>"

    start_indices = []
    end_indices = []

    index = 0
    while index < len(s):
        start_index = s.find(start_tag, index)
        if start_index == -1:
            break
        start_indices.append(start_index)
        index = start_index + len(start_tag)

    index = 0
    while index < len(s):
        end_index = s.find(end_tag, index)
        if end_index == -1:
            break
        end_indices.append(end_index + len(end_tag))
        index = end_index + len(end_tag)

    # 保留第一个和最后一个标签之间的内容
    if len(start_indices) <= 1 or len(end_indices) <= 1:
        return s

    for i in range(1, len(start_indices)-1):
        if i < len(end_indices):
            s = s[:start_indices[i]] + s[end_indices[i]:]


    return s
   

def _longest_common_prefix_len(a: str, b: str) -> int:
    """Return length of longest common prefix between a and b."""
    m = min(len(a), len(b))
    i = 0
    # compare characters until mismatch
    while i < m and a[i] == b[i]:
        i += 1
    return i


def compute_longest_shared_prefixes(strings: List[str], *, include_matches: bool = False) -> Dict[int, Dict]:
    """
    对于输入的字符串列表，计算每个字符串与其他字符串共享的最长前缀。

    返回一个字典，键为原始索引，值为一个字典，包含：
      - 'max_len': 最长共有前缀长度（int）
      - 'prefix': 最长共有前缀字符串（当 max_len>0 时，否则为 ""）
      - 'matches' (可选，当 include_matches=True 时返回): 与之共享该前缀的其他字符串的原始索引列表

    算法说明：先按字符串字典序对 (string, idx) 排序，最长共有前缀必出现在排序相邻项中，
    因此只需比较每个项与其前后邻居得到最长长度，然后以该长度从相邻位置收集所有匹配项。

    参数：
      - strings: List[str]
      - include_matches: bool, 是否返回匹配项索引（默认为 False）

    返回示例：
      {
        0: {'max_len': 3, 'prefix': 'app', 'matches': [1,2]},
        1: {'max_len': 0, 'prefix': ''}
      }
    """
    n = len(strings)
    # normalize to str and keep original indices
    strs = ["" if s is None else str(s) for s in strings]
    if n == 0:
        return {}

    arr = sorted(((s, i) for i, s in enumerate(strs)), key=lambda x: x[0])
    sorted_strs = [s for s, _ in arr]
    sorted_idx = [i for _, i in arr]

    # compute best lengths by comparing with neighbors
    best_len_sorted = [0] * n
    for i in range(n):
        cur = sorted_strs[i]
        best = 0
        if i > 0:
            best = max(best, _longest_common_prefix_len(cur, sorted_strs[i - 1]))
        if i < n - 1:
            best = max(best, _longest_common_prefix_len(cur, sorted_strs[i + 1]))
        best_len_sorted[i] = best

    # assemble result in original index order
    result: Dict[int, Dict] = {}
    for pos_in_sorted, orig_idx in enumerate(sorted_idx):
        max_len = best_len_sorted[pos_in_sorted]
        prefix = sorted_strs[pos_in_sorted][:max_len] if max_len > 0 else ""
        entry: Dict = {"max_len": max_len, "prefix": prefix}

        if include_matches and max_len > 0:
            # collect all adjacent items in sorted list that share this prefix
            matches: List[int] = []
            # left
            j = pos_in_sorted - 1
            while j >= 0 and sorted_strs[j].startswith(prefix):
                matches.append(sorted_idx[j])
                j -= 1
            # right
            j = pos_in_sorted + 1
            while j < n and sorted_strs[j].startswith(prefix):
                matches.append(sorted_idx[j])
                j += 1
            entry["matches"] = sorted(matches)

        result[orig_idx] = entry

    return result



def compute_longest_shared_prefixes_tokens(token_lists: List[List[int]], *, include_matches: bool = False) -> Dict[int, Dict]:
    """与 compute_longest_shared_prefixes 类似，但输入为 token 列表。

    参数：
      - token_lists: List[List[int]]
      - include_matches: bool, 是否返回匹配项索引（默认为 False）
    返回：
      - Dict[int, Dict]
    """
    n = len(token_lists)
    if n == 0:
        return {}

    arr = sorted(((tokens, i) for i, tokens in enumerate(token_lists)), key=lambda x: x[0])
    sorted_tokens = [tokens for tokens, _ in arr]
    sorted_idx = [i for _, i in arr]

    # compute best lengths by comparing with neighbors
    best_len_sorted = [0] * n
    for i in range(n):
        cur = sorted_tokens[i]
        best = 0
        if i > 0:
            best = max(best, _longest_common_prefix_len(cur, sorted_tokens[i - 1]))
        if i < n - 1:
            best = max(best, _longest_common_prefix_len(cur, sorted_tokens[i + 1]))
        best_len_sorted[i] = best

    # assemble result in original index order
    result: Dict[int, Dict] = {}
    for pos_in_sorted, orig_idx in enumerate(sorted_idx):
        max_len = best_len_sorted[pos_in_sorted]
        prefix = sorted_tokens[pos_in_sorted][:max_len] if max_len > 0 else []
        entry: Dict = {"max_len": max_len, "prefix": prefix}

        if include_matches and max_len > 0:
            # collect all adjacent items in sorted list that share this prefix
            matches: List[int] = []
            # left
            j = pos_in_sorted - 1
            while j >= 0 and sorted_tokens[j][:max_len] == prefix:
                matches.append(sorted_idx[j])
                j -= 1
            # right
            j = pos_in_sorted + 1
            while j < n and sorted_tokens[j][:max_len] == prefix:
                matches.append(sorted_idx[j])
                j += 1
            entry["matches"] = sorted(matches)

        result[orig_idx] = entry

    return result


class TrieNode:
    __slots__ = ("children", "request_idx")
    def __init__(self):
        self.children = {}  # token -> TrieNode
        self.request_idx = -1  # 该节点对应的请求序列索引

def process_in_order(token_lists: List[List[int]]) -> int:
    """
    按给定顺序处理序列，使用前缀缓存复用已出现的前缀，返回需要计算的 token 总数。
    等价于：对于每个序列，找到与之前所有序列共有的最长前缀长度 LMP，
    然后累加 len(seq) - LMP。

    使用 Trie 来逐步插入已经处理过的序列，时间复杂度 O(S)（S 为总 token 数）。
    """
    root = TrieNode()
    total = 0
    idx = 0 
    total_len = 0
    for seq in token_lists:
        node = root
        matched = 0
        # match as long as child exists
        for tok in seq:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break
        total += len(seq) - matched
        # insert remaining nodes
        for tok in seq[matched:]:
            newn = TrieNode()
            node.children[tok] = newn
            node = newn
        idx += 1
        total_len += len(seq)
        # if(idx%500 ==0):
            # print(f"Processed {idx} sequences, compute tokens {total}, total length {total_len} , cache ratio: {1 - total / total_len:.4f}")
    return total


def get_store_shared_tensor(data):
    #按顺序记录每个样本可以从前面出现样本中 最长复用的前缀长度以及是哪个样本 
    # 以及可以给后面还没出现样本提供多少个token的前缀
    shared_prefix_len = []
    store_for_sample_idx = [{} for _ in range(len(data))]
    store_for_sample_idx_ = [{} for _ in range(len(data))]

    # store_for_sample_idx[i] {request_idx: (start,end)} 
    shared_for_sample_idx = [{} for _ in range(len(data))]
    # shared_for_sample_idx[i] {request_idx: (start,end)} 
    root = TrieNode()
    for sample_idx, seq in enumerate(data):
        node = root
        matched = 0
        shared_request_idx = -1
        each_token_hit_req = []
        # match as long as child exists
        for tok in seq:
            if tok in node.children:
                node = node.children[tok]
                each_token_hit_req.append(node.request_idx)
                matched += 1
            else:
                break
        shared_prefix_len.append(matched)

        # insert remaining nodes
        for tok in seq[matched:]:
            newn = TrieNode()
            newn.request_idx = sample_idx
            node.children[tok] = newn
            node = newn

        # 处理 each_token_hit_req，对前缀进行划分
        
        for share_request_idx in set(each_token_hit_req):
            if share_request_idx == -1:
                continue
            # 找到该 request_idx 在 each_token_hit_req 中的连续区间
            start = None
            end = None
            for i, req_idx in enumerate(each_token_hit_req):
                if req_idx == share_request_idx:
                    if start is None:
                        start = i
                    end = i + 1  # end 是开区间
            if start is not None and end is not None:
                # 记录该样本可以从 share_request_idx 复用的前缀区间
                shared_for_sample_idx[sample_idx][share_request_idx] = (start, end)
                store_for_sample_idx[share_request_idx][sample_idx] = (start, end)
    # print("shared_for_sample_idx[30]:", shared_for_sample_idx[30])
    for sample_idx in range(len(data)):
        # 对 store_for_sample_idx[sample_idx] 保留最长的前缀区间 并记录最大的sample_idx
        base_address = 100000000
        end_address = 0
        last_sample_idx = 0
        for key in store_for_sample_idx[sample_idx].keys():
            start, end = store_for_sample_idx[sample_idx][key]
            base_address = min(base_address, start)
            end_address = max(end_address, end)
            last_sample_idx = max(last_sample_idx ,key)
        if(end_address > base_address):
            store_for_sample_idx_[sample_idx] = {
                "base_address": base_address,
                "end_address": end_address,
                "last_sample_idx": last_sample_idx
            }

    return  shared_prefix_len, store_for_sample_idx_, shared_for_sample_idx

def min_with_trie(token_lists: List[List[int]]) -> int:
    """
    全局最优（允许任意重排）情况下需要计算的最小 token 数。
    等价于把所有序列插入同一 Trie，新增的节点数就是需要计算的 token 数。
    """
    root = TrieNode()
    node_count = 0
    for seq in token_lists:
        node = root
        for tok in seq:
            if tok not in node.children:
                node.children[tok] = TrieNode()
                node_count += 1
            node = node.children[tok]
    return node_count


import heapq

def kk_partition(arr, order, n):
    """Karmarkar-Karp算法改进版"""
    new_order = [ [] for _ in range(len(arr))]
    if n <= 0 or n > len(arr):
        return []
    
    # 使用最大堆
    max_heap = [(-arr[i], i) for i in range(len(arr))]
    heapq.heapify(max_heap)
    
    # 初始化分组
    groups = [[] for _ in range(n)]
    sums = [0] * n
    
    # 每次取出最大的数，放入当前和最小的分组
    while max_heap:
        num, idx = heapq.heappop(max_heap)
        num = -num
        min_index = min(range(n), key=lambda i: sums[i])
        groups[min_index].append(num)
        sums[min_index] += num
        new_order[min_index].append(order[idx])
    return groups, sums, new_order


# random partition
def random_partition(arr, n):
    import random
    if n <= 0 or n > len(arr):
        return []
    
    groups = [[] for _ in range(n)]
    sums = [0] * n
    
    for num in arr:
        index = random.randint(0, n - 1)
        groups[index].append(num)
        sums[index] += num
    
    return groups, sums



# def partition_micro_batch(token_lists: List[List[int]], partition_n: int) -> List[List[List[int]]]:

#     # get the count 


#     def count_list_fun(token_lists: List[List[int]],  cache_list: TrieNode) -> int:
#         total = 0
#         result = [ ]
#         for seq in token_lists:
#             max_matched = 0
#             node = cache_list
#             # match as long as child exists
#             for tok in seq:
#                 if tok in node.children:
#                     node = node.children[tok]
#                     max_matched += 1
#                 else:
#                     break
#             total += len(seq) - max_matched
#             result.append(len(seq) - max_matched)
#             # insert remaining nodes
#             for tok in seq[max_matched:]:
#                 newn = TrieNode()
#                 node.children[tok] = newn
#                 node = newn

#         return result

#     count = process_in_order(token_lists)
#     # print(f"Total tokens to compute without partition: {count}")


#     average = count / partition_n
#     print(f"Average tokens per partition: {average}")
#     partitioned_token_lists = []
#     total_count = 0
#     cache_list = TrieNode()
#     for i in range(partition_n):
#         count_list = count_list_fun(token_lists, cache_list)
#         current_seq = []
#         current_count = 0
#         # 使当前的分区的count尽量接近average
#         for count_idx in range(len(token_lists)):
#             if(count_list[count_idx] + current_count <= average or i == partition_n -1):
#                 current_seq.append(token_lists[count_idx])
#                 current_count += count_list[count_idx]
#                 # 把当前序列加入cache_list
#                 for tok in token_lists[count_idx]:
#                     if tok not in cache_list.children:
#                         cache_list.children[tok] = TrieNode()
#                     cache_list = cache_list.children[tok]
#                 # remove the selected sequence
#                 token_lists[count_idx] = []
#                 # update count_list
#                 count_list = count_list_fun(token_lists, cache_list)
#             elif (count_list[count_idx] + current_count - average < average * 0.5):
#                 current_seq.append(token_lists[count_idx])
#                 current_count += count_list[count_idx]
#                 # 把当前序列加入cache_list
#                 for tok in token_lists[count_idx]:
#                     if tok not in cache_list.children:
#                         cache_list.children[tok] = TrieNode()
#                     cache_list = cache_list.children[tok]
#                 # remove the selected sequence
#                 token_lists[count_idx] = []
#                 # update count_list
#                 # count_list = count_list_fun(token_lists, cache_list)
#                 break
#         # average = (count - total_count) / (partition_n - i - 1) if (partition_n - i - 1) >0 else 0
#         # build the cache list
        
#         # update the token_lists and count_list to
#         # remove the selected sequences
#         new_token_lists = []
#         for count_idx in range(len(token_lists)):
#             if token_lists[count_idx] not in current_seq:
#                 new_token_lists.append(token_lists[count_idx])
#         token_lists = new_token_lists
#         total_count += current_count
#         partitioned_token_lists.append(current_seq)
#         print(f"Partition {i}: total tokens to compute: {current_count}")
            
#     print(f"Total tokens to compute with partition: {total_count}")
            
#     return partitioned_token_lists


def partition_micro_batch_cache_aware(token_lists: List[List[int]], partition_n: int) :

    total_count = process_in_order(token_lists)
    # print(f"Total tokens to compute without partition: {total_count}")
    average = total_count / partition_n
    # print(f"Average tokens per partition: {average}")
    partitioned_token_lists = []
    root = TrieNode()
    total = 0
    idx = 0 
    total_len = 0
    current_len = 0
    current_total_len = 0
    current_seq = []
    cache_ratio_list = []
    for seq in token_lists:
        node = root
        matched = 0
        # match as long as child exists
        for tok in seq:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break
        total += len(seq) - matched
        # insert remaining nodes
        for tok in seq[matched:]:
            newn = TrieNode()
            node.children[tok] = newn
            node = newn
        idx += 1
        current_len += len(seq) - matched
        current_seq.append(seq)
        current_total_len += len(seq)
        if(current_len >= average):
            partitioned_token_lists.append(current_seq)
            cache_ratio = 1 - current_len / current_total_len
            # print(f"Partition {len(partitioned_token_lists)-1}: total len: {current_total_len}, need to compute len: {current_len}, cache ratio: {cache_ratio:.4f}")
            current_seq = []
            current_len = 0
            current_total_len = 0
            cache_ratio_list.append(cache_ratio)
    # print("total count with partition:", total)
    return partitioned_token_lists, cache_ratio_list

def partition_micro_batch(token_lists: List[List[int]], partition_n: int ,max_token=2048) :

    total_count = sum([len(seq) for seq in token_lists])
    partition_n = total_count // max_token 
    # print(f"Total tokens to compute without partition: {total_count}")
    average = total_count / partition_n
    # print(f"Average tokens per partition: {average}")
    partitioned_token_lists = []
    current_len = 0
    current_seq = []
    remain = 0
    for seq in token_lists:
        current_len += len(seq) 
        current_seq.append(seq)
        if(current_len >= average+remain):
            partitioned_token_lists.append(current_seq)
            # print(f"Partition {len(partitioned_token_lists)}: need to compute len: {current_len}", "average+remain:", average+remain)
            current_seq = []
            remain = -current_len + average + remain
            current_len = 0

    print("nbs:", len(partitioned_token_lists))
    # exit()
    return partitioned_token_lists

def partition_micro_batch_token_level(token_lists: List[List[int]], partition_n: int ,max_token=2048) -> List[List[List[int]]]:
    total_count = process_in_order(token_lists)
    # return
    partition_n = total_count // max_token 
    print(f"Total tokens to compute without partition: {total_count}")
    average = total_count // partition_n
    print(f"Average tokens per partition: {average}")
    partitioned_token_lists = []
    root = TrieNode()
    total = 0
    idx = 0 
    current_len = 0
    current_total_len = 0
    current_seq = []
    cache_ratio_list = []
    while(idx < len(token_lists)):
        # print(f"Processing sequence {idx}")
        seq_idx = idx
        seq = token_lists[seq_idx]
        node = root
        matched = 0
        break_flag = False
        # match as long as child exists
        for tok in seq:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break
        total += len(seq) - matched
        # insert remaining nodes
        idx += 1
        current_len += len(seq) - matched
        current_total_len += len(seq)
        current_seq.append(seq)
        over_len = max(current_len - average, 0)
        # print(f"Sequence {seq_idx}: matched {matched}, seq[:matched]: {seq[:matched]}", f"seq[matched:]: {seq[matched:]}, over_len: {over_len}, insert {seq[matched: len(seq)-over_len]}")

        for tok in seq[matched: len(seq)-over_len]:
            newn = TrieNode()
            node.children[tok] = newn
            node = newn
        # print(f"After adding sequence {seq_idx}: matched {matched}")
        if(current_len >= average and len(partitioned_token_lists) < partition_n-1):
            over_len = current_len - average
            if over_len > 0 and len(seq) > over_len:
                current_seq[-1] = seq[:len(seq)-over_len-1]
                current_len -= over_len
                current_total_len -= over_len
                token_lists.insert(seq_idx+1, seq)
                token_lists[seq_idx] = seq[:len(seq)-over_len]
            partitioned_token_lists.append(current_seq)
            cache_ratio = 1 - current_len / current_total_len
            # print(f"Partition {len(partitioned_token_lists)-1}: current_seq : {current_seq}")
            # print(f"Partition {len(partitioned_token_lists)-1}: total len: {current_total_len}, need to compute len: {current_len}, cache ratio: {cache_ratio:.4f}")
            current_seq = []
            current_len = 0
            current_total_len = 0
            cache_ratio_list.append(cache_ratio)
        elif idx == len(token_lists) and len(current_seq) > 0:
            partitioned_token_lists.append(current_seq)
            cache_ratio = 1 - current_len / current_total_len
            # print(f"Partition {len(partitioned_token_lists)-1}: current_seq : {current_seq}")
            # print(f"Partition {len(partitioned_token_lists)-1}: total len: {current_total_len}, need to compute len: {current_len}, cache ratio: {cache_ratio:.4f}")
            current_seq = []
            current_len = 0
            current_total_len = 0
            cache_ratio_list.append(cache_ratio)
    print("nbs:", len(partitioned_token_lists))
    # exit()
    return partitioned_token_lists


def initialize_data():
    tokenized_data = torch.load("/workspace/heteflex/tools/token_1_id.pt")
    pipe_num = 2


    partition_n = pipe_num
    for current_pipe in range(pipe_num):
        uid_data_order = []
        uid_data_comp_count = []
        part_id  = current_pipe
        total_data_len = 0
        for uid in tokenized_data.keys():
            uid_data = []
            for tracj_id in tokenized_data[uid].keys():
                for turn in tokenized_data[uid][tracj_id].keys():
                    uid_data.append(tokenized_data[uid][tracj_id][turn])
                    total_data_len += 1
            uid_data_comp_count.append(process_in_order(uid_data))
            uid_data_order.append(uid)
        print("Total data length is ", total_data_len)
        kk_partitions, kk_sums, kk_order = kk_partition(uid_data_comp_count, uid_data_order, partition_n)
        print("Karmarkar-Karp Partitions sums for pipe ", current_pipe, " : ", kk_sums)
        # print(f"Processing partition {part_idx} with Karmarkar-Karp assignment")
        part_uids = [kk_order[current_pipe][i] for i in range(len(kk_order[current_pipe])) if uid_data_comp_count[uid_data_order.index(kk_order[current_pipe][i])] in kk_partitions[current_pipe]]
        distributed_data = []
        for uid in part_uids:
            for tracj_id in tokenized_data[uid].keys():
                for turn in tokenized_data[uid][tracj_id].keys():
                    distributed_data.append(tokenized_data[uid][tracj_id][turn])
        count = process_in_order(distributed_data)

        print("len of distributed data for pipe ", current_pipe, " is ", len(distributed_data))
        continue
        part_mbs = partition_micro_batch(distributed_data, partition_n=100)
        args.mbs_each_forward =  [ len(mb) for mb in part_mbs]

        gbs = 512 * 5 // pipe_num  # global batch size
        args.train_iters= (len(distributed_data) - 1) // gbs + 1
        args.nbs_each_iter = {}
        args.gbs_each_iter = []
        collective_mbs_data = []

        last_forward_id = 0
        forward_id = 0
        current_sample = 0
        for i in range(args.train_iters):
            if (i+1)*gbs <= len(distributed_data):
                args.gbs_each_iter.append(gbs)
            else:
                args.gbs_each_iter.append(len(distributed_data) - i*gbs)
            while(current_sample < args.gbs_each_iter[-1]):
                if forward_id >= len(args.mbs_each_forward) or current_sample + args.mbs_each_forward[forward_id] >= args.gbs_each_iter[-1]:
                    break
                current_sample += args.mbs_each_forward[forward_id]
                forward_id += 1
            args.nbs_each_iter[i] = forward_id - last_forward_id
            last_forward_id = forward_id
            current_sample = 0

        last_idx = 0
        for i in args.mbs_each_forward:
            start_idx = last_idx
            end_idx = min(last_idx + i, len(distributed_data))
            collective_mbs_data.append(distributed_data[start_idx:end_idx])
            last_idx = end_idx

        args.distributed_data = collective_mbs_data
        print("train iters = ", args.train_iters, "nbs_each_iter" ,args.nbs_each_iter )


def test():
    token =  torch.load(f"/workspace/heteflex/tools/token_1_id.pt")
    data = []
    uid_data_order = []
    uid_data_comp_count = []
    total_data = []
    sample_num_step = 0
    sample_num_token = 0
    data_token = []
    token_len = 0
    step_len = 0
    print("total len of uids is ", len(token.keys()))
    for uid in token.keys():
        uid_data = []
        for tracj_id in token[uid].keys():
            sample_num_token +=1
            for idx, turn in enumerate(token[uid][tracj_id].keys()):
                sample_num_step +=1
                data.append(token[uid][tracj_id][turn])
                uid_data.append(token[uid][tracj_id][turn])
                step_len += len(token[uid][tracj_id][turn])
                if(idx == len(token[uid][tracj_id].keys()) -1):
                    data_token.append(token[uid][tracj_id][turn])
                    token_len+= len(token[uid][tracj_id][turn])
        uid_data_comp_count.append(process_in_order(uid_data))
        uid_data_order.append(uid)

    token_count = process_in_order(data_token)
    step_compute_count = process_in_order(data)
    print(f"token count: {token_count}, step compute count: {step_compute_count}")
    print(f"avg token: {token_count/sample_num_token}, avg step: {step_compute_count/sample_num_step}")
    print(f"avg token len: {token_len/sample_num_token}", f"avg step len: {step_len/sample_num_step}")
if __name__ == "__main__":

    test()
    exit()
    # json_file =read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_no_think/1.jsonl")


    # token = load_tokenizer("Qwen/Qwen2.5-7B-Instruct",json_file)
    # # print("token [0]:",token[0])
    # res = compute_longest_shared_prefixes_tokens(token, include_matches=False)
    # # print("Result:",res)
    # total_len = 0
    # total_prex_len = 0
    # for i in range(len(token)):
    #     total_len += len(token[i])
    #     total_prex_len += res[i]['max_len']
    # print(f"total len: {total_len}, total prefix len: {total_prex_len} ,ratio: {total_prex_len/total_len}")


    # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_no_think/1.jsonl")
    # # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_cat_think/1.jsonl")
    # # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search/3.jsonl")
    # token = load_tokenizer("Qwen/Qwen2.5-7B-Instruct",test_data)
    # count = process_in_order(token)
    # total_len = 0
    # for i in range(len(token)):
    #     total_len += len(token[i])
    # cache_ratio = 1 - count / total_len
    # print(f"total len: {total_len}, need to compute len: {count}, cache ratio: {cache_ratio}")

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    # for j in range(1,2):
    #     test_data = read_json_file(f"/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_no_think/{j}.jsonl")
    #     # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_cat_think/1.jsonl")
    #     # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search/3.jsonl")
    #     token = [tokenizer.encode_plus(text, add_special_tokens=False)['input_ids'] for text in test_data]
    #     count = process_in_order(token)
    #     total_len = 0
    #     for i in range(len(token)):
    #         total_len += len(token[i])
    #     cache_ratio = 1 - count / total_len
    #     print(f"step {j}: total len: {total_len}, need to compute len: {count}, cache ratio: {cache_ratio}")


    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    # for j in range(1,2):
    #     test_data = read_json_file(f"/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_no_think/{j}.jsonl")
    #     # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_cat_think/1.jsonl")
    #     # test_data = read_json_file("/home/ymj/project/verl-agent/rollout_data_dir/grpo_search/3.jsonl")
    #     token = [tokenizer.encode_plus(text, add_special_tokens=False)['input_ids'] for text in test_data]
    #     save_token_path = f"/home/ymj/project/verl-agent/rollout_data_dir/grpo_search_no_think/token_{j}.pt"
    #     import torch
    #     torch.save(token, save_token_path)
    #     print(f"Saved tokens to {save_token_path}")
        
    #     loaded_tokens = torch.load(save_token_path)
    #     assert loaded_tokens == token
    #     print(f"Verified loaded tokens match original for step {j}")

    # initialize_data()
    # exit(0)


    partition_n = 2
    token =  torch.load(f"/workspace/heteflex/tools/token_1_id.pt")
    data = []



    uid_data_order = []
    uid_data_comp_count = []
    total_data = []
    print("total len of uids is ", len(token.keys()))
    for uid in token.keys():
        uid_data = []
        for tracj_id in token[uid].keys():
            for turn in token[uid][tracj_id].keys():
                data.append(token[uid][tracj_id][turn])
                uid_data.append(token[uid][tracj_id][turn])
        uid_data_comp_count.append(process_in_order(uid_data))
        uid_data_order.append(uid)



    print("len(uid_data_comp_count):", len(uid_data_comp_count))
    # random_partitions, random_sums = random_partition(uid_data_comp_count, partition_n)
    kk_partitions, kk_sums, kk_order = kk_partition(uid_data_comp_count, uid_data_order, partition_n)
    for i in range(partition_n):
        print("len(kk_partitions[", i, "]):", len(kk_partitions[i]))
    # print("Random Partitions:", random_sums)
    print("Karmarkar-Karp Partitions:", kk_sums)
    for part_idx in range(partition_n):
        # print(f"Processing partition {part_idx} with Karmarkar-Karp assignment")
        part_uids = [kk_order[part_idx][i] for i in range(len(kk_order[part_idx])) if uid_data_comp_count[uid_data_order.index(kk_order[part_idx][i])] in kk_partitions[part_idx]]
        print("len of uids in partition ", part_idx, " is ", len(part_uids))
        part_token_lists = []
        for uid in part_uids:
            for tracj_id in token[uid].keys():
                for turn in token[uid][tracj_id].keys():
                    part_token_lists.append(token[uid][tracj_id][turn])
        count = process_in_order(part_token_lists)
        total_len = 0
        for seq in part_token_lists:
            total_len += len(seq)
        cache_ratio = 1 - count / total_len
        # print(f"Karmarkar-Karp Partition {part_idx}: total len: {total_len}, need to compute len: {count}, cache ratio: {cache_ratio}")
        # print("len of token lists for pipe ", part_idx, " is ", len(part_token_lists))
        part_mbs, cache_ratio_list = partition_micro_batch_token_level(part_token_lists, partition_n=100)

        # print("part_mbs len for pipe ", part_idx, " is ", (part_mbs))
        forward_count_list = [ len(mb) for mb in part_mbs]
        actual_count = [ item*(1-cache_ratio_list[idx])  for idx, item in enumerate(forward_count_list)]

        # print(f"Micro-batch sizes for pipe {part_idx}:", forward_count_list)
        print(f"Actual compute counts for pipe {part_idx}:", actual_count)
        



            