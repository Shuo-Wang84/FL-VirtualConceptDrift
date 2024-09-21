import copy
import csv
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from scipy.special import gammaln
from scipy.stats import ks_2samp, kurtosis, skew
from scipy import stats
from scipy.stats import ttest_ind

from libcdd.unsupervised_drift.D3 import D3

"""Kolmogorov-Smirnov 检验 (KS-检验)：scipy.stats.ks_2samp 函数用于比较两个独立样本的分布是否相同。它可以检测两个样本之间的差异。

Anderson-Darling 检验：scipy.stats.anderson 函数执行 Anderson-Darling 检验，用于检测样本是否来自特定分布，例如正态分布。

Shapiro-Wilk 检验：scipy.stats.shapiro 函数用于检测一个样本是否来自正态分布。如果 p 值很小，就可能拒绝正态性假设。

Kruskal-Wallis 检验：scipy.stats.kruskal 函数用于比较多个独立样本的分布是否相同，但不要求样本来自正态分布。

Chi-Square 检验：scipy.stats.chisquare 函数用于检验观测值和预期值之间的差异，通常用于类别型数据的分布检验。

Mann-Whitney U 检验：scipy.stats.mannwhitneyu 函数用于比较两个独立样本的分布是否相同，但不要求样本来自正态分布。

Kolmogorov-Smirnov 二样本检验：scipy.stats.ks_2samp 函数用于比较两个样本的分布是否相同。

Chi-Square 独立性检验：scipy.stats.chi2_contingency 函数用于检验两个或多个分类变量之间的独立性。"""


# Define your MLP model 换一下 1、数据换一下hyperplane 可能不是模型问题 2、算法detect，维度升，分开好还是合并检测好,直接处理多维的检测方法和处理一维的检测方法对比。
class NewModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NewModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# #Define your new model class 这个模型不太好
# class NewModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NewModel, self).__init__()
#         # Define your own network architecture here
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x


# Load your data (assuming you have a function to read ARFF files)
class load_data(Dataset):
    def __init__(self, data_file):
        super(load_data, self).__init__()

        # Check the file extension
        if data_file.endswith('.csv'):
            self._load_csv(data_file)
        elif data_file.endswith('.arff'):
            self._load_arff(data_file)
        else:
            raise ValueError("Unsupported file format.")

    def _load_csv(self, data_file):
        # Load data from CSV file
        with open(data_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            data = list(csv_reader)
        # Extract features and labels from the loaded data
        self.features = []
        self.labels = []
        for line in data:
            self.features.append([float(f) for f in line[:-1]])
            self.labels.append(float(line[-1]))
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def _load_arff(self, data_file):
        # Load data from ARFF file
        with open(data_file, 'r') as f:
            lines = f.readlines()

        data_start = False
        data_lines = []
        for line in lines:
            if not data_start:
                if line.startswith('@data'):
                    data_start = True
            else:
                if not line.startswith('@'):
                    data_lines.append(line.strip())

        # Process data lines to extract features and labels
        data = [line.split(',') for line in data_lines]
        self.features = torch.tensor([list(map(float, row[:-1])) for row in data], dtype=torch.float64)
        self.labels = torch.tensor([float(row[-1]) for row in data], dtype=torch.float64)

    # Split data into 10 clients
    def split_data(data, num_clients=10, batch_size=100):
        # Initialize empty lists for each client
        clients_data = [[] for _ in range(num_clients)]
        data_streams = []

        # Distribute data points sequentially to clients
        for i, datapoint in enumerate(data):
            client_index = i % num_clients
            clients_data[client_index].append(datapoint)

        for i in range(num_clients):
            data_stream = DataLoader(clients_data[i], batch_size=batch_size, shuffle=False)
            data_streams.append((data_stream))

        # for i in range(len(data)):
        #     print("len(data)---", data[i])
        #
        # print("len(data_stream):",len(data_stream))
        return clients_data, data_streams

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FixedSizeWindow:
    def __init__(self, max_size):
        self.max_size = max_size
        self.W_data = []

    def add_data_stream(self, new_data_stream):
        # 将新的数据流添加到窗口
        self.W_data.append(new_data_stream)

        # 如果窗口大小超过最大值，删除最旧的数据流
        if len(self.W_data) > self.max_size:
            self.W_data.pop(0)  # 删除最旧的数据流

    def get_window_data(self):
        return self.W_data

    def __len__(self):
        return len(self.W_data)

    def __getitem__(self, key):
        return self.W_data[key]


# Training function for a local model
def train_local_model(model, train_data, local_epoch):
    # Define your loss function and optimizer
    # criterion = nn.NLLLoss()  # 这个损失函数不太好
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(local_epoch):
        inputs, labels = train_data
        model.train()
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Calculate accuracy
def calculate_accuracy(model, test_data):
    correct = 0
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        inputs, labels = test_data
        outputs = model(inputs)
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = (correct / len(inputs)) * 100.0
    return accuracy


# upload data into 10 clients
def upload_data(drift_type, num_clients, batch_size):
    # Initialize empty lists for each client
    clients_data = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        # print(f"i={i}")
        csv_name = f"{drift_type}"
        csv_name = f"data/vitual/{csv_name}_{i}.csv"
        # csv_name = f"E:\Program Files\PyCharm2021.2.2\PycharmProjects\pythonProject1\conceptdrift\generatedata/{csv_name}_{i}.csv"
        train_dataset = load_data(csv_name)
        for data_id, datapoint in enumerate(train_dataset):
            clients_data[i].append(datapoint)

    data_streams = []
    for i in range(num_clients):
        data_stream = DataLoader(clients_data[i], batch_size=batch_size, shuffle=False)
        data_streams.append((data_stream))

    return clients_data, data_streams


def driftDetection(window_data, n_feature):
    """使用D3方法,使用前一段和后一段的数据"""
    w = 200  # 你的窗口大小
    rho = 0.3  # 你的 rho 值
    auc = 0.6  # 你的 AUC 阈值
    D3_win = D3(w, rho, n_feature, auc)
    d_points = []
    d_points_y = []
    for d_point in window_data.get_window_data():
        # print("d_point=", d_point)
        d_points.append(np.array(d_point[0].cpu()))
        d_points_y.append(np.array(d_point[1].cpu()))
    # print("d_points",d_points)

    feature_data = np.concatenate(d_points, axis=0)
    feature_data_y = np.concatenate(d_points_y, axis=0)
    # 逐行提取并输出
    for row, y in zip(feature_data, feature_data_y):
        if D3_win.isEmpty():
            D3_win.addInstance(row, y)
        else:
            if D3_win.driftCheck():
                return True
            else:
                D3_win.addInstance(row, y)
    return False



def federated_stream_artificial():
    # 创建一个 FixedSizeWindow 对象列表，每个客户端一个，最大容纳 10 段数据流
    window_data_list = [FixedSizeWindow(4) for _ in range(num_clients)]
    drift_points = []  # 用于存储漂移点的列表

    # 人工数据集
    clients_data, data_streams = upload_data(drift_Flie_name, num_clients, batch_size)

    global_model = copy.deepcopy(model)
    local_model = [copy.deepcopy(model) for _ in range(num_clients)]
    global_test_acc = []

    for epoch in range(global_epochs):
        total_accuracy = 0.0
        local_models = []  # 不知道这样会不会把local_model[client_id]清除

        for client_id in range(num_clients):
            data_streams[client_id] = iter(data_streams[client_id])
            # print(f"data_streams[{client_id}]={data_streams[client_id]}")
            client_data = next(data_streams[client_id])
            client_data = [d.to(device) for d in client_data]
            # print(f"client_data[{client_id}]={client_data}")
            # print(f"historical_data[{client_id}]={historical_data}")

            # 将新的数据流添加到窗口
            window_data_list[client_id].add_data_stream(client_data)
            # print(f"window_data_list[{client_id}]={window_data_list[client_id]}")
            # Test the global model accuracy on each client's data
            accuracy = calculate_accuracy(local_model[client_id], client_data)
            # print(f"第{epoch}个全局模型对client_id", client_id, "的测试准确率= ", accuracy)
            total_accuracy += accuracy

            # 检测漂移
            ifdetection = driftDetection(window_data_list[client_id], n_feature)
            if ifdetection:
                # print(f"在epoch{epoch}客户端{client_id}数据流发生漂移！")
                drift_points.append((epoch, client_id))  # 记录漂移点

            train_local_model(local_model[client_id], client_data, local_epoch)
            local_models.append(local_model[client_id])

        # # Average local models into the global model
        # 将本地模型的参数平均到全局模型中

        for global_param, local_param_list in zip(global_model.parameters(),
                                                  zip(*[local_model[client_id].parameters() for
                                                        local_model[client_id] in local_models])):
            global_param.data = torch.mean(torch.stack(local_param_list), dim=0)

        # 在每个客户端上加载全局模型的参数
        for local_model[client_id] in local_models:
            local_model[client_id].load_state_dict(global_model.state_dict())

        # 全局准确率是所有客户端准确率的平均值，从epoch1开始是第1个全局模型参数上传给每个客户端后测试第2段数据流的准确率
        global_accuracy = total_accuracy / len(clients_data)

        # print(f"----------------------Epoch {epoch}: Global Model Accuracy = {global_accuracy}")
        global_test_acc.append(global_accuracy)

    # 画全局正确率图
    # global_test_accs = []
    # if len(global_test_accs) == 0:
    #     global_test_accs = global_test_acc
    # else:
    #     global_test_accs = np.array(global_test_accs)
    #     global_test_acc = np.array(global_test_acc)
    #     global_test_accs = global_test_accs + global_test_acc
    #     global_test_accs = global_test_accs.tolist()
    #
    # # 设置Matplotlib后端
    # # plt.switch_backend('TkAgg')  # 使用TkAgg后端
    # # 创建Figure
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(global_test_accs)), global_test_accs)
    # plt.xlabel('Time stamp')
    # plt.ylabel('Test Accuracy')
    # plt.ylim(0, 101)
    # plt.title('Global Test Accuracy')
    # plt.grid(True)
    # plt.savefig(f"./new-results/global_D3_alp{alpha}_{test_name}_acc.png")

    # 绘制漂移点图
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 5))

    if drift_points:
        epochs, client_ids = zip(*drift_points)
        plt.scatter(epochs, client_ids, c='red', marker='o', label='Drift Points')

    plt.xlabel('Time steps')
    plt.ylabel('Client ID')
    plt.title('Drift Points')
    plt.ylim(-0.2, 9.2)
    plt.xlim(0, global_epochs)

    # 保存漂移点图
    plt.savefig(f"./new-results/{id}_D3_alp{alpha}_{drift_Flie_name}.png")
    plt.show()
    plt.pause(0.1)


if __name__ == "__main__":
    global_epochs = 200

    local_epoch = 1
    num_clients = 10
    batch_size = 100
    gpu = 0
    sensitivity = 0.1
    Nmax = 10
    padding = 50
    # 设置显著性水平
    alpha = 0.05
    n_feature = 8
    id = 2  # 1：2D 2：Md 3：circle

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    print(device)
    model = NewModel(input_size=8, hidden_size=512, output_size=2)
    model = model.to(device)
    # test_name = f"virtual_circle_Asyn_g{global_epochs}_l{local_epoch}"
    drift_Flie_name = 'vitual_sudden_MD_Asyn'
    federated_stream_artificial()
