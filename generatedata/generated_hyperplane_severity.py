import csv
import random

import numpy as np
import pandas as pd
from skmultiflow.data import HyperplaneGenerator
import matplotlib.pyplot as plt

for j in range(10):
    # 创建超平面数据生成器
    generator = HyperplaneGenerator(random_state=j, n_features=2, n_drift_features=2, mag_change=0,
                                    noise_percentage=0,
                                    sigma_percentage=0)
    # 生成数据
    data = []
    labels = []
    data_squeeze = []
    num_samples = 30000 + j*1000
    for _ in range(num_samples):
        X, y = generator.next_sample()
        X_s = X.squeeze()
        data.append(X)
        labels.append(y)
        data_squeeze.append(X_s)

    data = np.array(data)
    labels = np.array(labels)

    # 创建数据帧并保存为CSV文件
    df = pd.DataFrame(data_squeeze, columns=generator.feature_names)
    df['target'] = labels
    df.to_csv(f'hyperplane_severity_all.csv', index=False)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], color='red', label='Class 0')
    plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], color='blue', label='Class 1')
    # 添加图例和标签
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Hyperplane Data Visualization')
    plt.legend()
    plt.grid(True)
    # 保存图片到文件
    plt.savefig('hyperplane_data_visualization.png')
    # 显示图形
    plt.show()

    # 读取生成的超平面数据集 CSV 文件
    df = pd.read_csv('hyperplane_severity_all.csv')
    # 将数据按类别分成两个列表
    class_0_data = df[df['target'] == 0]
    class_1_data = df[df['target'] == 1]
    # 随机打乱类别为0和1的样本
    class_0_data = class_0_data.sample(frac=1, random_state=42)
    class_1_data = class_1_data.sample(frac=1, random_state=42)

    # 将每个类别的样本按照特征值的大小排序
    class_0_data = class_0_data.sort_values(by=['att_num_0', 'att_num_1'])
    class_1_data = class_1_data.sort_values(by=['att_num_0', 'att_num_1'])

    # 将每个类别的样本分成三份
    num_samples = len(class_0_data)
    num_samples_per_partition = num_samples // 3
    print(f"num_samples_per_partition={num_samples_per_partition}")
    num_samples = len(class_1_data)
    num_samples_per_partition = num_samples // 3

    partitions_class_0 = [class_0_data[i:i + num_samples_per_partition] for i in
                          range(0, num_samples, num_samples_per_partition)]
    partitions_class_1 = [class_1_data[i:i + num_samples_per_partition] for i in
                          range(0, num_samples, num_samples_per_partition)]

    # 依次从两个类别中各抽取一份保存到数据文件中
    for i in range(3):
        # 从每个类别的三份样本中各抽取一份
        selected_data_class_0 = partitions_class_0[i]
        selected_data_class_1 = partitions_class_1[i]

        # 将抽取的样本合并成一个数据集
        selected_data = pd.concat([selected_data_class_0, selected_data_class_1])

        # 将抽取的样本保存到数据文件中
        selected_data.to_csv(f'selected_data_partition_{i + 1}.csv', index=False)

        # 读取数据文件
        df = pd.read_csv(f'selected_data_partition_{i + 1}.csv')

        # 提取特征和标签
        features = df[['att_num_0', 'att_num_1']]
        labels = df['target']

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(features[labels == 0]['att_num_0'], features[labels == 0]['att_num_1'], color='red',
                    label='Class 0')
        plt.scatter(features[labels == 1]['att_num_0'], features[labels == 1]['att_num_1'], color='blue',
                    label='Class 1')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title('Data Visualization')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'selected_data_partition_{i + 1}_visualization.png')
        plt.show()

    # 1. 读取CSV文件并加载数据
    with open('selected_data_partition_1.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # 2. 打乱数据顺序
    random.shuffle(data)

    # 3. 将打乱后的数据重新写入CSV文件中
    with open('selected_data_partition_1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # 1. 读取CSV文件并加载数据
    with open('selected_data_partition_2.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # 2. 打乱数据顺序
    random.shuffle(data)

    # 3. 将打乱后的数据重新写入CSV文件中
    with open('selected_data_partition_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    data1 = []
    data2 = []

    with open('selected_data_partition_1.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        data1 = list(reader)

    with open('selected_data_partition_2.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)

    # 2. 合并两个文件的数据
    merged_data = data1 + data2

    # 3. 将合并后的数据写入一个新的CSV文件中
    merged_file = f'hyperplane_Severity_1-2_{j}.csv'

    with open(merged_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(merged_data)
