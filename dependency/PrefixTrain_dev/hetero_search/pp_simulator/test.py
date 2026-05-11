from simulator import InterleavedOneFOneBGenerator, OperationExecutor, OneFOneBGenerator, EagerOneFOneBGenerator
from operations import Config, HyperConfig
import itertools

def permute(n):
    return list(itertools.permutations(range(n)))

def sort_dict_by_value(d):
    # 按照字典的值进行排序
    sorted_items = sorted(d.items(), key=lambda item: item[1])
    # 将排序后的项转换回字典
    sorted_dict = {k: v for k, v in sorted_items}
    return sorted_dict

def get_time (fwd,bwd,a,p,m):

    stage_time_list = []
    for i in range(p):
        stage_time = fwd[i] +bwd[i]
        if(i>0):
            stage_time+=a[i-1][i]
        if(i<p-1):
            stage_time+=a[i][i+1]
        stage_time_list.append(stage_time)
    sum_stage_time =  sum(stage_time_list)
    max_stage_time = max(stage_time_list)
    # max_idx = stage_time_list.index(max_stage_time)

    # return sum_stage_time + (m-1)*max_stage_time - (p-max_idx-1)*fwd[max_idx]
    return sum_stage_time + (m-1)*max_stage_time

def test_1():
    order = permute(4)

    pp = 4
    vp = 1


    # # (pp,vp)
    # fwd_per_stage = [7, 6, 6.5, 8]
    # bwd_per_stage = [13, 12, 13, 14]
    fwd_per_stage = [1, 3, 4, 5]
    bwd_per_stage = [2, 6, 8, 7.5]
    # fwd_per_stage = [4, 3, 5, 1]
    # bwd_per_stage = [8, 6, 10, 2]





    result_list = {}
    for tuple in order:
        fwd = []
        bwd = []
        for i in  tuple:
            fwd.append(fwd_per_stage[i])
            bwd.append(bwd_per_stage[i])
        # (pp, pp)
        a = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        # a = [
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [1, 0, 0, 0]
        # ]

        r = [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]

        config = HyperConfig(f=fwd, b=bwd, a=a, r=r, p=4, m=6, v=1, c=0, overlap_c=True)
        og = OneFOneBGenerator(config)
        operations = og.generate()
        executor = OperationExecutor(config, operations)
        executor.execute()
        ans = executor.makespan()
        
        ans_2 = get_time (fwd,bwd,a,4,6)
        # if(ans ==  99 or ans ==114):
        #     executor.export_trace(f"test {ans}.json")
        result_list[tuple] = ans 
        print(f"ans {ans} ,  ans2 {ans_2}")
    sort_dict = sort_dict_by_value(result_list)

    for key in sort_dict.keys():
        print(f"{key}: {sort_dict[key]}")


def test_2():
    r = [
    [True, True, False, False],
    [True, True, False, False],
    [False, False, True, True],
    [False, False, True, True],
    ]
    comm_table =  [0, 24.233164062499995, 0, 0], [24.233164062499995, 0, 47.40423828125, 0], [0, 47.40423828125, 0, 47.40423828125], [0, 0, 47.40423828125, 0]
    fwd_per_stage=[372.0654400000009, 384.1515250000012, 302.228797, 7.109094]
    bwd_per_stage=[717.62299, 745.2708500000009, 622.4290669999995, 126.635313]

    config_ = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=comm_table, r=r, p=4, m=128, v=1, c=0, overlap_c=True)
    og = OneFOneBGenerator(config_)
    operations = og.generate()
    executor = OperationExecutor(config_, operations)
    executor.execute()
    ans = executor.makespan()
    print(ans)

def test_3():

    fwd_per_stage = [1, 3, 4]
    bwd_per_stage = [2, 6, 8]
    num_stage = len(fwd_per_stage)
    order = permute(num_stage)
    m = 600
    a = [ [ ] for _ in range(num_stage)]
    r = [ [ True for __ in range(num_stage)] for _ in range(num_stage)]

    a = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    result_list = {}
    for tuple in order:
        fwd = []
        bwd = []
        for i in  tuple:
            fwd.append(fwd_per_stage[i])
            bwd.append(bwd_per_stage[i])

        config = HyperConfig(f=fwd, b=bwd, a=a, r=r, p=num_stage, m=m, v=1, c=0, overlap_c=True)
        og = OneFOneBGenerator(config)
        operations = og.generate()
        executor = OperationExecutor(config, operations)
        executor.execute()
        ans = executor.makespan()

        ans_2 = get_time (fwd,bwd,a,num_stage,m)
        # executor.export_trace(f"test{ans}.json")
        result_list[tuple] = ans 
        print(f"ans {ans} ,  ans2 {ans_2}")


    sort_dict = sort_dict_by_value(result_list)
    for key in sort_dict.keys():
        print(f"{key}: {sort_dict[key]}")



def test_4():

    fwd_per_stage = [[1,1], [3,3], [4,4]]
    bwd_per_stage = [[2,2], [6,6] ,[8,8]]
    num_stage = len(fwd_per_stage)
    order = permute(num_stage)
    m = 600
    a = [ [ ] for _ in range(num_stage)]
    r = [ [ True for __ in range(num_stage)] for _ in range(num_stage)]

    a = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


    config = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=a, r=r, p=num_stage, m=m, v=2, c=0, overlap_c=True)
    og = InterleavedOneFOneBGenerator(config)
    operations = og.generate()
    executor = OperationExecutor(config, operations)
    executor.execute()
    ans = executor.makespan()

    # ans_2 = get_time (fwd,bwd,a,num_stage,m)
    # executor.export_trace(f"test{ans}.json")
    # result_list[tuple] = ans 
    # print(f"ans {ans} ,  ans2 {ans_2}")


    sort_dict = sort_dict_by_value(result_list)
    for key in sort_dict.keys():
        print(f"{key}: {sort_dict[key]}")

def test_5():
    pp = 4
    vp = 2

    # (pp,vp)
    fwd_per_stage = [[4, 4], [4, 5], [4, 4], [4, 5]]
    bwd_per_stage = [[8, 8], [8, 10], [8, 8], [8, 10]]

    # (pp, pp)
    a = [
        [0, 1, 10, 10],
        [1, 0, 10, 10],
        [10, 10, 0, 1],
        [10, 10, 1, 0]
    ]

    r = [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]

    config = HyperConfig(f=fwd_per_stage, b=bwd_per_stage, a=a, r=r, p=4, m=16, v=2, c=0, overlap_c=True)
    og = InterleavedOneFOneBGenerator(config)
    operations = og.generate()
    executor = OperationExecutor(config, operations)
    executor.execute()
    ans = executor.makespan()
    print(ans)


if __name__ =="__main__":
    test_3()