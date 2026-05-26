import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.colors as mcolors
import glob
# Rebuild the font cache
# matplotlib.font_manager._rebuild()

# Read the CSV file

# Create a timeline plot
fig, ax = plt.subplots()



# Define colors for different names
'''
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
    add_to_logging('batch-generator')'''

colors = { 
    'forward-compute': '#385294', #'182,196,229'
    'forward-recv': 'red',
    'forward-send': 'purple',
    'forward-send-backward-recv': 'orange',
    # 'forward-send-forward-recv': 'brown',
    # 'forward-backward-send-forward-backward-recv': 'pink',
    'backward-compute': '#aacf8f',
    'backward-recv': 'black',
    'backward-send': 'gray',
    'backward-send-forward-recv': 'cyan',
    # 'backward-send-backward-recv': 'magenta',
    # 'backward-params-all-reduce': 'olive',
    # 'backward-embedding-all-reduce': 'lime',
    # 'optimizer': 'teal',
    # 'batch-generator': 'navy'
}

# 填充花纹
hatch_par  = {
    'forward-compute': None,
    'backward-compute': None,
    'forward-recv': '\\',
    'forward-send': '//',
    'forward-send-backward-recv': '//',
    'backward-recv': '//',
    'backward-send': '\\',
    'backward-send-forward-recv': '\\',

}

# Plot each operation as a line on the timeline



def read_csv(file_pattern):
    data = []
    # 定义CSV文件的文件模式
    # 获取匹配模式的文件列表
    file_list = sorted(glob.glob(file_pattern))
    stage_data = {}
    # 初始化一个字典来存储每个stage的数据
    for file_name in file_list:
        stage = int(file_name.split('_stage')[1].split('.')[0][0])
        with open(file_name, 'r') as file:
            reader = csv.DictReader(file)
            data = []
            for row in reader:
                data.append(row)
            stage_data[stage] = data

    min_start_time =float(stage_data[0][0]['start_time'])
    for stage in stage_data:
        # Convert start_time and end_time to relative time (starting from 0)
        for row in stage_data[stage]: 
            min_start_time = min(min_start_time, float(row['start_time']))


    for stage in stage_data:
        # Convert start_time and end_time to relative time (starting from 0)
        for row in stage_data[stage]:
            row['start_time'] = (float(row['start_time']) - min_start_time)
            row['end_time'] = (float(row['end_time']) - min_start_time)

        # 按照start_time排序
        stage_data[stage] = sorted(stage_data[stage], key=lambda x: x['start_time'])

    return stage_data

def draw(stage_data,start_idx =0, ylabel = [20, 22,23]):
    save_total_time = 130
    for idx, stage in enumerate(sorted(stage_data.keys(), reverse=True)):
        print(idx, stage)
        for row in stage_data[stage]:
            start_time = row['start_time']
            if(start_time > save_total_time):
                continue
            end_time = row['end_time']
            name = row['name']
            if(colors.get(name) is None):
                continue
            # 减小两个stage之间的间隔
            if("send" in name or "recv" in name):
                alpha = 0.5
            else:
                alpha = 1
            ax.barh(stage+start_idx, end_time - start_time, left=start_time, color=colors[name], alpha=alpha, label=name, hatch=hatch_par.get(name, None))
        
    # Set the y-axis labels








if __name__ == "__main__":

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_22-20-23_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 0 )

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_20-23-22_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 4)

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_20-22-23_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 8)

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_22-23-20_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 12)

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_23-22-20_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 16)

    file_pattern = '/workspace/aceso/logs-large/our_test/runtime/gpt/1_3B/total_csv/gpt1_3B_n2_g2_23-20-22_stage*_*.csv'
    stage_data = read_csv(file_pattern)
    draw(stage_data, 20)


    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_title('Operation Timeline')

    # Create a legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #legend 位置放在外面
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6)
    plt.yticks([0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22], ['22','20','23','20','23','22','20','22','23','22','23','20','23','22','20','23','20','22'])



    # Show the plot
    # plt.show()

    # save the plot 设置大小

    fig.set_size_inches(18, 9)
    plt.savefig('./pictures/final.png', dpi=2048)