from typing import List
from itertools import chain


class ListElement:
    def __init__(self, value, next):
        self.value = value
        self.next = next
    def nth(self, n):
        o = self
        i = 0
        while i < n and o.next is not None:
            o = o.next
            i += 1
        return o

def init(multiset):
    multiset.sort() # ensures proper non-increasing order
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)

def visit(h):
    """Converts our bespoke linked list to a python list."""
    o = h
    l = []
    while o is not None:
        l.append(o.value)
        o = o.next
    return l

def permutations(multiset):
    """Generator providing all multiset permutations of a multiset."""
    h, i, j = init(multiset)
    yield visit(h)
    while j.next is not None or j.value < h.value:
        if j.next is not None and i.value >= j.next.value:
            s = j
        else:
            s = i
        t = s.next
        s.next = t.next
        t.next = h
        if t.value < h.value:
            i = t
        j = i.next
        h = t
        yield visit(h)

def gen_dgroups_recursive(num_stages: int, num_gpus: int, group_shapes: List):
    def f(current_sum, stage_idx, curr_sol, prev_shape_idx):
        # filtering
        if group_shapes[-1] * (num_stages - stage_idx) < num_gpus - current_sum:
            # max gpu < total gpu
            return
        if group_shapes[0] * (num_stages - stage_idx) > num_gpus - current_sum:
            # min gpu > total gpu
            return

        if stage_idx >= num_stages:
            if len(curr_sol) == num_stages and current_sum == num_gpus:
                yield curr_sol
            return

        for i in range(max(0, prev_shape_idx), len(group_shapes)):
            possible_gpu_num = group_shapes[i]
            if possible_gpu_num + current_sum > num_gpus:
                break
            my_sol = curr_sol + [possible_gpu_num]
            yield from f(current_sum + possible_gpu_num, stage_idx + 1, my_sol, i)

    for idx, possible_gpu_num in enumerate(group_shapes):
        yield from f(possible_gpu_num, 1, [possible_gpu_num], idx)

def permute(s, max_permute_len):
    def find_num_min(m, groups):
        for i, e in enumerate(groups):
            if m != e:
                return i + 1
        return len(groups)

    # grouping
    groups = [(e,) for e in s]
    # Key idea 2: limit the permutation length
    curr_permute_len = len(groups)
    num_reduce = curr_permute_len - max_permute_len
    while num_reduce > 0:
        min_group_size = sum(groups[0])
        num_min_groups = find_num_min(groups[0], groups)
        # Example:
        #  - device groups: (1), (1), (1), (1), (1), (1), (2)
        #  - max_permute_len: 6
        #  - generated: (1,1), (1,1), (1,1), (2)
        if num_min_groups // 2 > num_reduce:
            num_reduce = num_min_groups // 2

        # Merge the two smallest groups
        merged_groups = []
        for i in range(0, len(groups), 2):
            curr_reduce_num = i // 2
            if num_reduce <= curr_reduce_num:
                # End of merge
                merged_groups.extend(groups[i:])
                break
            if i + 1 >= len(groups):
                merged_groups.append(groups[i])
            else:
                if sum(groups[i]) == min_group_size and sum(groups[i]) == sum(groups[i + 1]):
                    merged_group = tuple(groups[i] + groups[i + 1])
                    merged_groups.append(merged_group)
                else:
                    merged_groups.append(groups[i])
                    merged_groups.append(groups[i + 1])

        groups = merged_groups
        if num_reduce == len(groups) - max_permute_len:
            # we can't reduce anymore
            break
        num_reduce = len(groups) - max_permute_len

    # print("Merged groups:", groups)
    perms = permutations(groups)
    return perms

def gen_device_group_shapes(num_gpus: int) -> List[int]:
    # print(f"num_gpus: {num_gpus}")
    group_shapes = []
    i = 0
    while 2 ** i <= num_gpus:
        group_shapes.append(2 ** i)
        i += 1
    # print(f"group_shapes: {group_shapes}")
    return group_shapes

def get_device_group_list(num_stages, num_gpus ,variance = 1.0, max_permute_len = 4 ,min_gpus_per_stage=1):
    min_group_stage = max(num_gpus // num_stages, num_stages // num_gpus)
    # min_group_stage = min_gpus_per_stage #TODO 
    # print(f"min_group_stage: {min_group_stage}")
    # print(f"variance: {variance}")
    min_group_stage *= variance
    group_shapes = gen_device_group_shapes(num_gpus)
    group_shapes = [s for s in group_shapes if s >= min_group_stage]
    device_groups = []
    # print(f"num_stages: {num_stages}, num_gpus: {num_gpus}, group_shapes: {group_shapes}")
    for s in gen_dgroups_recursive(num_stages, num_gpus, group_shapes):
        perm_s = permute(s, max_permute_len)
        for perm in perm_s:
            perm_ss = list(chain(*perm))
            device_groups.append(perm_ss)

    device_groups = [list(x) for x in set(tuple(x) for x in device_groups)]
    # print(f"device_groups: {device_groups}")
    return device_groups







def gen_dgroups_for_stages_with_variance(num_stages: int, num_gpus: int, group_shapes: List[int], variance: float,
                                         max_permute_len: int) -> List:
    # Key idea 1: Limit the size of device group
    min_group_stage = max(num_gpus // num_stages, num_stages // num_gpus)
    min_group_stage *= variance
    group_shapes = [s for s in group_shapes if s >= min_group_stage]

    device_groups = []
    for s in gen_dgroups_recursive(num_stages, num_gpus, group_shapes):
        perm_s = permute(s, max_permute_len)
        for perm in perm_s:
            perm_ss = list(chain(*perm))
            device_groups.append(perm_ss)

    return device_groups

from itertools import combinations_with_replacement, permutations

def find_combinations(n, m, min_value,max_value):
    # 生成所有可能的2的幂次方且大于min_value
    powers_of_2 = [2**i for i in range(m.bit_length()) if (2**i>=min_value and 2**i<=max_value )]
    result = set()
    for comb in combinations_with_replacement(powers_of_2, n):
        if sum(comb) == m:
            # 对组合进行排列并去重
            for perm in permutations(comb):
                result.add(perm)
    result = [list(comb) for comb in result]
    return result

import itertools

def find_combinations_v1(n, m, min_value, max_value):
    possible_numbers = []
    x = 1  # 2^0
    while x <= max_value:
        if x >= min_value:
            possible_numbers.append(x)
        x *= 2
    
    if not possible_numbers or n <= 0:
        return []
    
    all_combos = itertools.product(possible_numbers, repeat=n)
    # print(all_combos)
    result = [list(combo) for combo in all_combos if sum(combo) == m]
    
    return result



import itertools

def find_combinations_v2(n, m, min_value, max_value, clist):
    # 检查clist的总和是否等于m
    if sum(clist) != m:
        return []
    
    # 生成所有可能的2的幂次方数，且在[min_value, max_value]范围内
    possible_numbers = []
    x = 1  # 2^0
    while x <= max_value:
        if x >= min_value:
            possible_numbers.append(x)
        x *= 2
    
    # 处理边界情况
    if not possible_numbers or n <= 0 or len(clist) == 0:
        return []
    
    # 生成所有可能的n元组，并过滤总和为m的组合
    all_combos = itertools.product(possible_numbers, repeat=n)
    filtered_combos = (combo for combo in all_combos if sum(combo) == m)
    
    # 检查每个组合是否符合clist的分割条件
    valid_combos = []
    for combo in filtered_combos:
        target_index = 0
        current_sum = 0
        valid = True
        for num in combo:
            if target_index >= len(clist):
                valid = False
                break
            current_sum += num
            if current_sum == clist[target_index]:
                target_index += 1
                current_sum = 0
            elif current_sum > clist[target_index]:
                valid = False
                break
        # 所有元素处理完毕后，必须所有目标都完成且current_sum为0
        if valid and target_index == len(clist) and current_sum == 0:
            valid_combos.append(list(combo))
    
    return valid_combos


def find_powers_of_two(n, m, k, current_sum=0, current_list=[], results=[]):
    # 如果已经选择了m个数，并且总和为n，加入结果
    if len(current_list) == m:
        if current_sum == n:
            # 排序后去重
            current_list.sort()
            if current_list not in results:
                results.append(current_list[:])
        return
    
    # 从2的幂次方开始选择，确保每个数都大于k
    power = 1
    while power <= n:
        if power > k:
            find_powers_of_two(n, m, k, current_sum + power, current_list + [power], results)
        power *= 2




# Input n and m
if __name__ == "__main__":
    # Input from the user
    # 查找并打印组合

    # for i in range(1,16):
    #     combinations_0 = find_combinations_v1(i,2048,2048//i,2048//i)
    #     print(len(combinations_0))
    #     for comb in combinations_0:
    #         print(comb) 
    

    # combinations = find_combinations_v2(6,32,2,8,[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    # # print("=====")
    # for comb in combinations:
    #     print(comb) 
    # print(f" { combinations == combinations_0}")

    # combinations = get_device_group_list(35, 280, 1, 1, 1)
    # print(combinations)

    # for comb in combinations:
    #     print(comb)

    # n = 14  # 总和
    # m = 5   # 需要的个数
    # k = 0   # 每个数必须大于k
    # results = []
    # find_powers_of_two(n, m, k, results=results)
    # print(results)
    # 输出所有结果
    # for res in results:
    #     print(res)


    #=    
    num_gpus =256
    for num_stages in range(2,16+1):
        device_group_list = find_combinations_v1(num_stages,num_gpus, 8, num_gpus//num_stages)
        if(device_group_list!=[]):
            print(device_group_list)