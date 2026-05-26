

'''
op_name,forward-compute,backward-compute,input_size,output_size,weights,activations,fwd_reserved,bwd_reserved
encoder-embedding,93.162,754.982,0.008,4.000,104.000,6.000,14.000,100.000
enc-1st-layernorm,41.646,168.896,4.000,8.000,0.000,4.016,0.000,20.000
enc-attention-qkv,153021.598,349.760,8.000,16.000,6.000,12.000,20.000,88.000
enc-attention-score,301.319,276.572,16.000,136.000,0.000,128.000,128.000,256.000
enc-attention-softmax,161.451,226.796,136.000,136.000,0.000,128.000,0.000,384.000
enc-attention-dropout,240.445,204.813,136.000,136.000,0.000,192.000,0.000,512.000
enc-attention-context,124.675,355.053,136.000,8.000,0.000,4.000,0.000,148.000
enc-attention-dense,54.353,184.679,8.000,8.002,2.000,4.000,0.000,20.000
enc-post-attention-dropout,62.734,124.496,8.002,4.000,0.000,6.000,0.000,36.000
enc-2nd-layernorm,40.966,117.868,4.000,8.000,0.000,4.016,0.000,20.000
enc-MLP-GEMM-1,86.695,177.884,8.000,20.008,8.000,16.000,0.000,68.000
enc-MLP-gelu,46.945,151.181,20.008,20.000,0.000,16.000,0.000,128.000
enc-MLP-GEMM-2,140.518,158.918,20.000,8.002,8.000,4.000,0.000,36.000
enc-post-MLP-dropout,57.596,78.881,8.002,4.000,0.000,6.000,0.000,36.000
final-layernorm,42.003,396.502,4.000,4.000,0.000,4.016,0.000,20.000
gpt-post-process,3648.001,2889.848,4.000,0.000,100.000,400.025,199.975,0.000
'''

#读取cvs文件的数据
import pandas as pd
import numpy as np
import json

def read_csv(file_path):
    return pd.read_csv(file_path)



if __name__ == '__main__':
    device_name = "A100"

    device_name_list= ["A100","RTX3090"]
    model_size_list = ["350M","1_3B"]
    mbs_list = [1,2,4,8]
    tp_list = [1,2]
    for device_name in device_name_list:
        for model_size in model_size_list:
            for tp in tp_list:
                for mbs in mbs_list:
                    
                    file_name = f"/workspace/aceso/profiler/profiled-time-hete/{device_name}/gpt_{model_size}/gpt_{model_size}_mbs{mbs}_tp{tp}_algo0.csv"
                    data = read_csv(file_name)
                    first_layer_compute_time = data['forward-compute'][0]+data['backward-compute'][0]
                    encoder_layer_compute_time = 0
                    for i in range(1, 13):
                        encoder_layer_compute_time += data['forward-compute'][i]+data['backward-compute'][i]
                    last_layer_compute_time = data['forward-compute'][14]+data['backward-compute'][14]+data["forward-compute"][15]+data["backward-compute"][15]

                    print("first_layer_compute_time:", first_layer_compute_time)
                    print("encoder_layer_compute_time:", encoder_layer_compute_time)
                    print("last_layer_compute_time:", last_layer_compute_time)

                    # 保存到json文件
                    save_data={"forward_backward_time_ms":0,"layer_compute_total_ms":[]}
                    save_data["forward_backward_time_ms"]=(first_layer_compute_time+encoder_layer_compute_time*24+last_layer_compute_time)/1000
                    save_data["layer_compute_total_ms"].append(first_layer_compute_time/1000)
                    for i in range(24):
                        save_data["layer_compute_total_ms"].append(encoder_layer_compute_time/1000)
                    save_data["layer_compute_total_ms"].append(last_layer_compute_time/1000)
                    
                    with open(f"/workspace/aceso/profile/profile_{model_size}/"+f"DeviceType.{device_name}_tp{tp}_bs{mbs}.json", 'r') as json_file:
                        data_json = json.load(json_file)
                        data_json["execution_time"]["forward_backward_time_ms"]=save_data["forward_backward_time_ms"]
                        data_json["execution_time"]["layer_compute_total_ms"]=save_data["layer_compute_total_ms"]
                    
                    with open(f"/workspace/aceso/profile/profile_{model_size}/"+f"DeviceType.{device_name}_tp{tp}_bs{mbs}.json", 'w') as json_file:
                        json.dump(data_json, json_file, indent=4)

