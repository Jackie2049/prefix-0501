# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from aceso_cost_model import read_profiled_time, get_memory_v3
from aceso_utils import *



gpt_configs = {
    "350M": (24, 2048, 1024, 1024*4, 16, 1024//16, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 2048*4, 32, 2048//32, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 2560*4, 32, 2560//32, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 4096*4, 32, 4096//32, 51200, "fp16"),
    "13B": (40, 2048, 5120, 5120*4, 40, 5120//40, 51200, "fp16"),
    "scale-layer": (1, 2048, 512, 512*4, 8, 512//8, 51200, "fp16")
}

device_config = [
                    {'host':'127.0.0.21','rank':0,'ID':"A100_1",'avaliable_memory':81920,'compute_capability':8.0,'FP32':312}, #FP32: 312 TFLOPS
                    {'host':'127.0.0.21','rank':0,'ID':"A100_2",'avaliable_memory':81920,'compute_capability':8.0,'FP32':312},
                    {'host':'127.0.0.21','rank':0,'ID':"A100_3",'avaliable_memory':81920,'compute_capability':8.0,'FP32':312}, #FP32: 312 TFLOPS
                    {'host':'127.0.0.21','rank':0,'ID':"A100_4",'avaliable_memory':81920,'compute_capability':8.0,'FP32':312},
                    {'host':'127.0.0.22','rank':1,'ID':"RTX3090_1",'avaliable_memory':24576,'compute_capability':8.6,'FP32':35.6864},
                    {'host':'127.0.0.22','rank':1,'ID':"RTX3090_2",'avaliable_memory':24576,'compute_capability':8.6,'FP32':35.6864},
                    {'host':'127.0.0.23','rank':2,'ID':"RTX3090_1",'avaliable_memory':24576,'compute_capability':8.6,'FP32':35.6864},
                    {'host':'127.0.0.23','rank':2,'ID':"RTX3090_2",'avaliable_memory':24576,'compute_capability':8.6,'FP32':35.6864}
                    ]

class TaskGraph:
    
    def __init__(self,stage_index,num_stages_behind,num_gpus,ops,tp_size,dp_size,algo,base_batch_size):
            super().__init__()
            self.stage_index = stage_index
            self.num_stages_behind = num_stages_behind
            self.num_gpus = num_gpus
            self.ops = ops
            self.tp_size = tp_size
            self.dp_size = dp_size
            self.algo = algo
            self.base_batch_size = base_batch_size
            self.memory = self.compute_taskGraph_memory()
            self.num_flop = self.compute_taskGraph_flops()
    
    def get_taskGraph_index(self):
        return self.stage_index
    
    def get_taskGraph_memory(self):
        return self.memory
    
    def get_taskGraph_num_gpus(self):
        return self.num_gpus
    
    def get_taskGraph_ops(self):
        return self.ops
    
    def get_taskGraph_base_batch_size(self):
        return self.base_batch_size
    
    def get_taskGraph_num_flop(self):
        return self.num_flop
    
    def get_taskGraph_dp_size(self):
        return self.dp_size
    
    def get_taskGraph_tp_size(self):
        return self.tp_size
    
    def get_tp_degree(self):
        return max(self.tp_size)
    
    def get_dp_degree(self):
        return max(self.dp_size)
    
    def compute_taskGraph_flops(self):
        num_flop = 0
        num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype= gpt_configs[args.model_size]
        for op in self.ops:
            if op == 'encoder-embedding':
                num_flop += 2 * self.base_batch_size * seq_len * hidden_size * vocab_size
            elif op == 'enc-1st-layernorm':
                num_flop +=  9 * self.base_batch_size * seq_len * hidden_size
            elif op == 'enc-attention-qkv':
                num_flop += 6 * self.base_batch_size * seq_len * hidden_size * hidden_size 
            elif op == 'enc-attention-score':
                num_flop += 2 * self.base_batch_size * seq_len * seq_len * hidden_size
            elif op == 'enc-attention-softmax':
                num_flop += 4 * self.base_batch_size * num_attention_heads * seq_len * seq_len
            elif op == 'enc-attention-dropout':
                num_flop += self.base_batch_size * num_attention_heads * seq_len * seq_len
            elif op == 'enc-attention-context':
                num_flop += 2 * self.base_batch_size * seq_len * seq_len * hidden_size
            elif op == 'enc-attention-dense':
                num_flop += 2 * self.base_batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-post-attention-dropout':
                num_flop += self.base_batch_size * seq_len * hidden_size
            elif op == 'enc-2nd-layernorm':
                num_flop += 9 * self.base_batch_size * seq_len * hidden_size
            elif op == 'enc-MLP-GEMM-1':
                num_flop += 8 * self.base_batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-MLP-gelu':
                num_flop +=  20 * self.base_batch_size * seq_len * hidden_size 
            elif op == 'enc-MLP-GEMM-2':
                num_flop += 8 * self.base_batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-post-MLP-dropout':
                num_flop += self.base_batch_size * seq_len * hidden_size
            elif op == 'final-layernorm':
                num_flop += 9 * self.base_batch_size * seq_len * hidden_size
            elif op == 'gpt-post-process':
                num_flop += 2 * self.base_batch_size * seq_len * hidden_size * vocab_size
            else:
                num_flop += 0 
        return num_flop
    
    def compute_taskGraph_memory(self):
        mbs_list = [self.base_batch_size//self.dp_size[j] for j in range(len(self.ops))] 
        memory_weights, inputs, activations = get_memory_v3(self.ops, mbs_list, self.tp_size, self.algo)     
        memory_gradients = memory_weights
        memory_main_params = memory_weights * args.memory_main_params
        memory_optimizer = memory_weights * args.memory_optimizer 
        memory_activations = (inputs + activations) * self.num_stages_behind
        return memory_weights + memory_gradients + memory_main_params + memory_optimizer + memory_activations
    
    def compute_deviceGroup_taskGraph_flops(self,batch_size):
        num_flop = 0
        num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype= gpt_configs[args.model_size]
        for op in self.ops:
            if op == 'encoder-embedding':
                num_flop += 2 * batch_size * seq_len * hidden_size * vocab_size
            elif op == 'enc-1st-layernorm':
                num_flop +=  9 * batch_size * seq_len * hidden_size
            elif op == 'enc-attention-qkv':
                num_flop += 6 * batch_size * seq_len * hidden_size * hidden_size 
            elif op == 'enc-attention-score':
                num_flop += 2 * batch_size * seq_len * seq_len * hidden_size
            elif op == 'enc-attention-softmax':
                num_flop += 4 * batch_size * num_attention_heads * seq_len * seq_len
            elif op == 'enc-attention-dropout':
                num_flop += batch_size * num_attention_heads * seq_len * seq_len
            elif op == 'enc-attention-context':
                num_flop += 2 * batch_size * seq_len * seq_len * hidden_size
            elif op == 'enc-attention-dense':
                num_flop += 2 * batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-post-attention-dropout':
                num_flop += batch_size * seq_len * hidden_size
            elif op == 'enc-2nd-layernorm':
                num_flop += 9 * batch_size * seq_len * hidden_size
            elif op == 'enc-MLP-GEMM-1':
                num_flop += 8 * batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-MLP-gelu':
                num_flop +=  20 * batch_size * seq_len * hidden_size 
            elif op == 'enc-MLP-GEMM-2':
                num_flop += 8 * batch_size * seq_len * hidden_size * hidden_size
            elif op == 'enc-post-MLP-dropout':
                num_flop += batch_size * seq_len * hidden_size
            elif op == 'final-layernorm':
                num_flop += 9 * batch_size * seq_len * hidden_size
            elif op == 'gpt-post-process':
                num_flop += 2 * batch_size * seq_len * hidden_size * vocab_size
            else:
                num_flop += 0 
        return num_flop
    
    
    def compute_deviceGroup_taskGraph_memory(self,mirco_batch_size):
        
        ops_num = len(self.ops)
        dp_size = [1] * ops_num
        tp_size = [1] * ops_num
        algo = [0] * ops_num
        
        total_memory = 0
        batch_sizes = self.decompose_into_powers_of_two(mirco_batch_size)
        for batch_size in batch_sizes:
            mbs_list = [batch_size//dp_size[j] for j in range(ops_num)]    
            memory_weights, inputs, activations = get_memory_v3(self.ops, mbs_list, tp_size, algo)     
            memory_gradients = memory_weights
            memory_main_params = memory_weights * args.memory_main_params
            memory_optimizer = memory_weights * args.memory_optimizer 
            memory_activations = (inputs + activations) * self.num_stages_behind
            total_memory += memory_weights + memory_gradients + memory_main_params + memory_optimizer + memory_activations
        return  total_memory / len(batch_sizes)
    
    def decompose_into_powers_of_two(self,n):
        powers = []
        power = 0
        while n > 0:
            if n % 2 == 1:  
                powers.append(2 ** power)
            n //= 2  
            power += 1
        return powers
    def print_taskGraph(self):
        print('stage_index: {}'.format(self.stage_index))
        print('num_stages_behind: {}'.format(self.num_stages_behind))
        print('num_gpus: {}'.format(self.num_gpus))
        print('ops: {}'.format(self.ops))
        print('tp_size: {}'.format(self.tp_size))
        print('dp_size: {}'.format(self.dp_size))
        print('algo: {}'.format(self.algo))
        print('base_batch_size: {}'.format(self.base_batch_size))
        print('memory: {}'.format(self.memory))
        print('num_flop: {}'.format(self.num_flop))
        print("------------------------------------------------------------------")

class TaskGraphGroup:
    def __init__(self):
        super().__init__()
        self.taskGraphs = []
    
    def add_taskGraph(self,taskGraph):
        if not self.is_exist(taskGraph.get_taskGraph_index()):
            self.taskGraphs.append(taskGraph)
    
    def is_exist(self,index):
        for taskGraph in self.taskGraphs:
            if taskGraph.get_taskGraph_index() == index:
                return True
            else:
                continue
        
        return False
    
    def get_taskGraphs(self):
        return self.taskGraphs
        
    def taskGraph_sort_by_memory(self):
        return sorted(self.taskGraphs,key=lambda taskGraph:taskGraph.get_taskGraph_memory() * taskGraph.get_taskGraph_num_gpus(),reverse=True)
        
    def print_taskGraph(self):
        for graph in self.taskGraphs:
            graph.print_taskGraph()
            
class Device:
    def __init__(self,config):
        super().__init__()
        
        self.host = config['host']
        self.rank = config['rank']
        self.ID = config['host'] + '/' + config['ID']
        self.avaliable_memory = config['avaliable_memory']
        self.compute_capability = config['compute_capability']
        self.FP32 = config['FP32']
        self.ops = []
        self.batch_size = 0
    
    def get_device_host(self):
        return self.host
    
    def get_device_memory(self):
        return self.avaliable_memory
    
    def get_device_ID(self):
        return self.ID
    
    def get_device_FP32(self):
        return self.FP32
    
    def set_ops(self,ops):
        self.ops = ops
    
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
        
    def print_device(self):
        print('host: {}'.format(self.host))
        print('rank: {}'.format(self.rank))
        print('ID: {}'.format(self.ID))
        print('avaliable_memory: {}'.format(self.avaliable_memory))
        print('compute_capability: {}'.format(self.compute_capability))
        print('compute_capability: {}'.format(self.compute_capability))
        print('ops: {}'.format(self.ops))
        print('batch_size: {}'.format(self.batch_size))
        print("------------------------------------------------------------------")
        
class DeviceGroup:
    def __init__(self,deviceGroup_ID, tag_dp, tag_tp,batch_size):
        self.ID = deviceGroup_ID
        self.tag_dp = tag_dp
        self.tag_tp = tag_tp
        self.devices = []
        self.batch_size = batch_size
        
   
    
    def add_deviceGroup(self,deviceGroup):
        self.devices.append(deviceGroup)
        
    def add_device(self,device):
        self.devices.extend(device)
        
    def set_devices(self,taskGraph):
        if  not self.tag_dp:
            for device in self.devices:
                device.set_ops(taskGraph.get_taskGraph_ops())
                device.set_batch_size(self.batch_size)
                
    def set_deviceGroup_batch_size(self,batch_size):
        self.batch_size = batch_size
        
    def get_deviceGroup_devices(self):
        return self.devices   
    def get_tp_deviceGroup_batch_size(self):
        return self.batch_size
    
    def get_tp_deviceGroup_memory(self):
        if not self.tag_dp:
            return sum([ device.get_device_memory() for device in self.devices])
        
    def compute_tp_deviceGroup_fp32_sum(self):
        if not self.tag_dp:
            return sum([ device.get_device_FP32() for device in self.devices]) / len(self.devices)
    
    def compute_dp_deviceGroup_fp32_sum(self):
        total = 0
        for tp_deviceGroup in self.devices:
            total += sum([ device.get_device_FP32() for device in tp_deviceGroup.get_deviceGroup_devices()]) / len(tp_deviceGroup.get_deviceGroup_devices())
        
        return total 
            
    def print_deviceGroup_dp(self):
        print('deviceGroup_ID: {}'.format(self.ID))
        print('tag_dp: {}'.format(self.tag_dp))
        print('tag_tp: {}'.format(self.tag_tp))
        print('batch_size: {}'.format(self.batch_size))
        for deviceGroup_tp in self.devices:
            deviceGroup_tp.print_deviceGroup_tp();
        print("------------------------------------------------------------------")
    
    def print_deviceGroup_tp(self):
        print('deviceGroup_ID: {}'.format(self.ID))
        print('tag_dp: {}'.format(self.tag_dp))
        print('tag_tp: {}'.format(self.tag_tp))
        print('batch_size: {}'.format(self.batch_size))
        for device in self.devices:
            device.print_device()
        print("------------------------------------------------------------------")


class VirtualDevice:
    def __init__(self,taskGraph_ID,dp_deviceGroup,taskGraph):
        super().__init__()
        self.ID = taskGraph_ID
        self.dp_deviceGroup = dp_deviceGroup
        self.taskGraph = taskGraph
    
        
       
    
    def get_dp_deviceGroup(self):
        return self.dp_deviceGroup
    
    def get_taskGraph(self):
        return self.taskGraph
    def get_virtural_device_ID(self):
        return self.ID
    
    # def compute_device_fp32_sum(self):
    #     return sum([device.get_device_FP32() for device in self.devices])
    

    def print_virtual_device(self):
        print('ID: {}'.format(self.ID))
        print('devices: ')
        self.dp_deviceGroup.print_deviceGroup_dp()
        # print('taskGraph:')
        # self.dp_deviceGroup.print_taskGraph()
        
        print("------------------------------------------------------------------")
           
class Cluster:
    def __init__(self,device_configs):
        super().__init__()
        self.work_host = []
        self.master = ''
        self.avaliable_devices = []
        self.used_devices = []
        self.virtual_devices = []
        
        for config in device_configs:
            self.work_host.append(config['host'])
            self.master = config['host'] if config['rank'] == 0 else self.master
            device = Device(config)
            self.avaliable_devices.append(device)
        self.work_host = list(set(self.work_host))
        
    
    
    def memory_constraint_load_balancing(self):
        if len(self.virtual_devices) != 0:
            for virtual_device in self.virtual_devices:
                load_ratios = []
                mem_utils = []
                flop_utils = []
                oom_tp_deviceGroups = []
                free_tp_deviceGroups = []

                total_DF = virtual_device.get_dp_deviceGroup().compute_dp_deviceGroup_fp32_sum()
                dp_deviceGroup = virtual_device.get_dp_deviceGroup()
                taskGraph = virtual_device.get_taskGraph()
                tp_deviceGroup = dp_deviceGroup.get_deviceGroup_devices()

                if len(tp_deviceGroup) <=1:
                    continue
                else:
                    for i in range(len(dp_deviceGroup.get_deviceGroup_devices())):

                        load_ratios.append(tp_deviceGroup[i].compute_tp_deviceGroup_fp32_sum() / total_DF)
                        mem_utils.append((load_ratios[i] * taskGraph.compute_deviceGroup_taskGraph_memory(tp_deviceGroup[i].get_tp_deviceGroup_batch_size())) / tp_deviceGroup[i].get_tp_deviceGroup_memory())
                        flop_utils.append((load_ratios[i] * taskGraph.compute_deviceGroup_taskGraph_flops(tp_deviceGroup[i].get_tp_deviceGroup_batch_size())) / tp_deviceGroup[i].compute_tp_deviceGroup_fp32_sum())
                        if mem_utils[i] > 1 :
                            oom_tp_deviceGroups.append(i)
                        else:
                            free_tp_deviceGroups.append(i)
                
                print(load_ratios)
                print(mem_utils)
                print(flop_utils)
                print(oom_tp_deviceGroups)
                print(free_tp_deviceGroups)
                
                self.shift_load(load_ratios,mem_utils,flop_utils,oom_tp_deviceGroups,free_tp_deviceGroups,taskGraph,tp_deviceGroup)

    
    def shift_load(self,load_ratios,mem_utils,flop_utils,oom_tp_deviceGroups,free_tp_deviceGroups,taskGraph,tp_deviceGroup):
        
        while len(oom_tp_deviceGroups)!=0 and len(free_tp_deviceGroups)!=0:
            peak_tp_deviceGroup_index = self.argmax(oom_tp_deviceGroups,mem_utils)
            valley_tp_deviceGroup_index = self.argmin(free_tp_deviceGroups,mem_utils,flop_utils)
            peak_tp_deviceGroup = tp_deviceGroup[peak_tp_deviceGroup_index]
            valley_tp_deviceGroup = tp_deviceGroup[valley_tp_deviceGroup_index]
            
            peak_tp_deviceGroup_load_ratios = load_ratios[peak_tp_deviceGroup_index]
            valley_tp_deviceGroup_load_ratios = load_ratios[valley_tp_deviceGroup_index]
            
            
            peak_tp_deviceGroup_batch_size = peak_tp_deviceGroup.get_tp_deviceGroup_batch_size()
            if peak_tp_deviceGroup_batch_size <=1:
                oom_tp_deviceGroups = []
                free_tp_deviceGroups = []
            else:
                min_peak_tp_deviceGroup_shift_batch_size = 0
                for i in range(1,peak_tp_deviceGroup_batch_size):
                    mem_util = (peak_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_memory(peak_tp_deviceGroup_batch_size - i)) / peak_tp_deviceGroup.get_tp_deviceGroup_memory()
                    if mem_util <=1:
                        min_peak_tp_deviceGroup_shift_batch_size = i
                        break
                    else:
                        continue
                if min_peak_tp_deviceGroup_shift_batch_size == 0:
                    oom_tp_deviceGroups.remove(peak_tp_deviceGroup_index)
                else:
                    
                    valley_tp_deviceGroup_batch_size = valley_tp_deviceGroup.get_tp_deviceGroup_batch_size()
                    max_valley_tp_deviceGroup_get_batch_size = 0
                    for i  in range(1,peak_tp_deviceGroup_batch_size):
                        mem_util = (valley_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_memory(valley_tp_deviceGroup_batch_size + i)) / valley_tp_deviceGroup.get_tp_deviceGroup_memory()
                        if mem_util >1:
                            max_valley_tp_deviceGroup_get_batch_size = i
                            break
                        else:
                            if i== peak_tp_deviceGroup_batch_size -1:
                                max_valley_tp_deviceGroup_get_batch_size = i
                            continue
                        
                    if max_valley_tp_deviceGroup_get_batch_size < min_peak_tp_deviceGroup_shift_batch_size:
                        
                        free_tp_deviceGroups.remove(valley_tp_deviceGroup_index)
                        
                    elif peak_tp_deviceGroup_batch_size - max_valley_tp_deviceGroup_get_batch_size >=1:
                        peak_tp_deviceGroup.set_deviceGroup_batch_size(peak_tp_deviceGroup_batch_size - max_valley_tp_deviceGroup_get_batch_size)
                        valley_tp_deviceGroup.set_deviceGroup_batch_size(valley_tp_deviceGroup_batch_size + max_valley_tp_deviceGroup_get_batch_size)
                        mem_utils[valley_tp_deviceGroup_index] = (valley_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_memory(valley_tp_deviceGroup.get_tp_deviceGroup_batch_size())) / valley_tp_deviceGroup.get_tp_deviceGroup_memory()
                        flop_utils[valley_tp_deviceGroup_index] = (valley_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_flops(valley_tp_deviceGroup.get_tp_deviceGroup_batch_size())) / valley_tp_deviceGroup.compute_tp_deviceGroup_fp32_sum()
                        oom_tp_deviceGroups.remove(peak_tp_deviceGroup_index)
                    else:
                        peak_tp_deviceGroup_shift_batch_size = peak_tp_deviceGroup_batch_size - 1
                        peak_tp_deviceGroup.set_deviceGroup_batch_size(peak_tp_deviceGroup_batch_size - peak_tp_deviceGroup_shift_batch_size)
                        valley_tp_deviceGroup.set_deviceGroup_batch_size(valley_tp_deviceGroup_batch_size + peak_tp_deviceGroup_shift_batch_size)
                        mem_utils[valley_tp_deviceGroup_index] = (valley_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_memory(valley_tp_deviceGroup.get_tp_deviceGroup_batch_size())) / valley_tp_deviceGroup.get_tp_deviceGroup_memory()
                        flop_utils[valley_tp_deviceGroup_index] = (valley_tp_deviceGroup_load_ratios * taskGraph.compute_deviceGroup_taskGraph_flops(valley_tp_deviceGroup.get_tp_deviceGroup_batch_size())) / valley_tp_deviceGroup.compute_tp_deviceGroup_fp32_sum()
                        oom_tp_deviceGroups.remove(peak_tp_deviceGroup_index)
                        
                
            
    def argmax(self,oom_devices,mem_utils):
        oom_devices_memory = [mem_utils[pos] for pos in oom_devices]
        peak_device_memory  = max(oom_devices_memory)
        return oom_devices[oom_devices_memory.index(peak_device_memory)]

    def argmin(self,free_devices,mem_utils,flop_utils):
        free_devices_flop = [flop_utils[pos] for pos in free_devices]
        valley_devices_flop = min(free_devices_flop)
        valley_devices_flop_index = [free_devices[pos] for pos, value in enumerate(free_devices_flop) if value == valley_devices_flop]
        if len(valley_devices_flop_index)<=1:
            return free_devices[free_devices_flop.index(valley_devices_flop)]
        else:
            valley_devices_memory = [mem_utils[pos] for pos in valley_devices_flop_index]
            valley_device_memory = min(valley_devices_memory)
            return valley_devices_flop_index[valley_devices_memory.index(valley_device_memory)]
    
    def find_device_index(self,deviceList,tag_device):
        for i in range(len(deviceList)):
            if deviceList[i].get_device_ID() == tag_device.get_device_ID():
                return i
            else:
                continue
        return -1
    
        
        stage_deviceGroup =[] 
        
        taskGraphs = taskGroup.get_taskGraphs()
        for i,taskGraph in enumerate(taskGraphs):
            deviceGroup = self.create_taskGraph_deviceGroup(taskGraph)
            
            if deviceGroup is None:
                if i > 0:
                    pre_deviceGroup = stage_deviceGroup[i-1]
                    
            else:
                stage_deviceGroup.append(deviceGroup)    
    
    def create_taskGraph_deviceGroup(self,taskGraph):
        tp_degree = taskGraph.get_tp_degree()
        dp_degree = taskGraph.get_dp_degree()
        num_gpus = taskGraph.get_taskGraph_num_gpus()
        memory = taskGraph.get_taskGraph_memory()
        taskGraph_ID = taskGraph.get_taskGraph_index()
        base_batch_size = taskGraph.get_taskGraph_base_batch_size()
        dp_deviceGroup = DeviceGroup(taskGraph_ID,True,False,base_batch_size)
        
        if tp_degree > 1 and dp_degree > 1:
            devices = self.get_avaliable_device(tp_degree*dp_degree,memory)
            if devices is None:
                return None 
            micro_batch_size = base_batch_size // dp_degree
            for i in range(dp_degree):
                if i == 0:
                    micro_batch_size += base_batch_size % dp_degree
                deviceGroup_ID = 'stage-' + str(taskGraph_ID) + '/' + 'dp/' + str(i)
                tp_deviceGroup = DeviceGroup(deviceGroup_ID,False,True,micro_batch_size)
                tp_deviceGroup.add_device(devices[i*tp_degree:(i+1)*tp_degree])
                tp_deviceGroup.set_devices(taskGraph)
                
                self.add_used_device(devices)
                self.remove_used_device_from_avalibale_devices()
                
                dp_deviceGroup.add_deviceGroup(tp_deviceGroup)
            
        elif tp_degree > 1:
                devices = self.get_avaliable_device(tp_degree,memory)
                if devices is None:
                    return None 
                deviceGroup_ID = 'stage-' + str(taskGraph_ID) + '/' + 'dp/0'
                tp_deviceGroup = DeviceGroup(deviceGroup_ID,False,True,base_batch_size)
                tp_deviceGroup.add_device(devices)
                tp_deviceGroup.set_devices(taskGraph)
                # tp_deviceGroup.print_deviceGroup_tp()
                
                self.add_used_device(devices)
                self.remove_used_device_from_avalibale_devices()
                dp_deviceGroup.add_deviceGroup(tp_deviceGroup)
        elif dp_degree > 1:
            
            devices = self.get_avaliable_device(tp_degree*dp_degree,memory)
            if devices is None:
                    return None 
            micro_batch_size = base_batch_size // dp_degree
            for i in range(dp_degree):
                if i == 0:
                    micro_batch_size += base_batch_size % dp_degree
                
                deviceGroup_ID = 'stage-' + str(taskGraph_ID) + '/' + 'dp/' + str(i)
                tp_deviceGroup = DeviceGroup(deviceGroup_ID,False,True,micro_batch_size)
                tp_deviceGroup.add_device(devices[i*tp_degree:(i+1)*tp_degree])
               
                tp_deviceGroup.set_devices(taskGraph)
                # tp_deviceGroup.print_deviceGroup_tp()
                
                self.add_used_device(devices)
                self.remove_used_device_from_avalibale_devices()
                dp_deviceGroup.add_deviceGroup(tp_deviceGroup)
        else:
                devices = self.get_avaliable_device(num_gpus,memory)
                if devices is None:
                    return None 
                deviceGroup_ID = 'stage-' + str(taskGraph_ID) + '/' + 'dp/0'
                tp_deviceGroup = DeviceGroup(deviceGroup_ID,False,False,base_batch_size)
                tp_deviceGroup.add_device(devices)
                tp_deviceGroup.set_devices(taskGraph)
                
                self.add_used_device(devices)
                self.remove_used_device_from_avalibale_devices()
                dp_deviceGroup.add_deviceGroup(tp_deviceGroup)
        
        return dp_deviceGroup
            
    def get_avaliable_device(self,num_gpus,memory):
        device_within_node = self.get_avaliable_device_within_node(num_gpus,memory)
        if device_within_node is not None:
            return device_within_node
        else:
            if len(self.avaliable_devices) < num_gpus:
                return None
            else:
                avaliable_devices = self.avaliable_devices.copy()
                sort_avaliable_device = sorted(avaliable_devices,key=lambda device:device.get_device_memory(),reverse=True)
                num_gpus_devices = sort_avaliable_device[:num_gpus]
                device_memory = sum(device.get_device_memory() for device in num_gpus_devices)
                if device_memory < memory * num_gpus :
                    return None
                else:
                    return num_gpus_devices
          
                 
    def get_avaliable_device_within_node(self,num_gpus,memory):
       
        node_devices = []
        for host in self.work_host:
            node_device = {"devices":[],'memory':0}
            for device in self.avaliable_devices:
                if str(device.get_device_host()) == str(host):
                    node_device['devices'].append(device)
            if len(node_device['devices']) >= num_gpus:
                node_device['devices'].sort(key=lambda device:device.get_device_memory(),reverse=True)
                node_device['devices'] = node_device['devices'][:num_gpus]
                node_device['memory'] = sum(device.get_device_memory() for device in node_device['devices'])
                node_devices.append(node_device)
            else:
                continue
        if len(node_devices)==0:
            return None
        else:
            node_max_memory =  max(node_devices,key=lambda node: node['memory'])
            if node_max_memory['memory'] < memory * num_gpus :
                return None
            else:
                return node_max_memory['devices']
    
    def get_max_memory_avaliable_device(self):
            return max(self.avaliable_devices,key=lambda device:device.get_device_memory())
        
    def get_avaliable_device_num(self):
        return len(self.avaliable_devices)
    
    def get_virtural_device(self):
        return self.virtual_devices
    
    def remove_used_device_from_avalibale_devices(self):
        for used_device in self.used_devices:
            device_index = self.find_device_index(self.avaliable_devices,used_device)
            if device_index != -1:
                self.avaliable_devices.pop(device_index)
            
    def add_used_device(self,devices):
            self.used_devices.extend(devices)
        
    def add_virtual_device(self,virtualDevice):
        self.virtual_devices.append(virtualDevice)
    
    def print_cluster(self):
        
        print('avaliable_devices: {}'.format([device.get_device_ID() for device in self.avaliable_devices]))
        print('used_devices: {}'.format([device.get_device_ID() for device in self.used_devices]))
        
        print("------------------------------------------------------------------")
    
    def get_physical_device(self):
        if len(self.virtual_devices) != 0:
            devices = []
            self.virtual_devices.sort(key=lambda virtualDevice: virtualDevice.get_virtural_device_ID())
            for i,virtualDevice in enumerate(self.virtual_devices):
                devices.append([])
                for tp_deviceGroup in virtualDevice.get_dp_deviceGroup().get_deviceGroup_devices():
                    
                    devices[i].append([device.get_device_ID() for device in tp_deviceGroup.get_deviceGroup_devices()])
            return devices
        else:
            return []
        
args = parse_args()


read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)



def generate_config_whale(full_op_list, tp_degree, dp_degree, layer_partirion, micro_bs, global_bs):
    
    num_ops_per_stage = []
    for index,num_layers in enumerate(layer_partirion):
        if index == 0:
            num_ops_per_stage.append(1 + (num_layers * 13))
        elif index == len(layer_partirion) - 1:
            num_ops_per_stage.append(2 + (num_layers * 13))
        else :
            num_ops_per_stage.append(num_layers * 13)
    
    tp_size_list = []
    dp_size_list = []
    
    for i in range(len(num_ops_per_stage)):
        tp_size_list += [tp_degree[i] for _ in range(num_ops_per_stage[i])]
        dp_size_list += [dp_degree[i] for _ in range(num_ops_per_stage[i])]
        
    
    recompute_ops = [0 for _ in range(len(full_op_list))]
    algo_list = [0 for _ in range(len(full_op_list))]

    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)
    return initial_config


def run_search():
    device = []
    cluster = Cluster(device_config)
    taskGroup = TaskGraphGroup()
    full_op_list = get_full_op_list(args)  
    
    #  op tp dp layers mirco global_batch_size
    config = generate_config_whale(full_op_list,[2,2,2],[2,1,1],[21,6,5],2,1024 )  # 根据args生成一个config
   
    for stage in config.stages:
        taskGraph = TaskGraph(stage.index,stage.num_stages_behind,stage.num_gpus,stage.ops,stage.tp_size,stage.dp_size,stage.algo,config.micro_bs)
        taskGroup.add_taskGraph(taskGraph)
    
    sorted_taskGraphs = taskGroup.taskGraph_sort_by_memory()
    
    # taskGroup.print_taskGraph()
    count  = 0
    for item in sorted_taskGraphs:
       
        devices = cluster.create_taskGraph_deviceGroup(item)
        
        
        if devices is not None:
            virtual_device = VirtualDevice(item.get_taskGraph_index(),devices,item)
            cluster.add_virtual_device(virtual_device)
        else:
            cluster = None
            taskGroup = None  
            break
        
    if cluster is not None :
        cluster.memory_constraint_load_balancing()
        device = cluster.get_physical_device()
    
    print(device)



    
        
if __name__=='__main__':
    run_search()
