import copy
import csv
import math
from libcdd.data_distribution_based import *
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from skmultiflow.drift_detection import DDM, ADWIN
from skmultiflow.drift_detection import KSWIN
from collections import OrderedDict
from scipy.special import gammaln
from scipy.stats import ks_2samp, kurtosis, skew
from scipy import stats
from scipy.stats import ttest_ind
from libcdd.data_distribution_based.kdqtree import KDQTree

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
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



# class NewModel(torch.nn.Module):
#     '''
#     Synthetic dataset models, optimized for periodic functions
#     '''
#
#     def __init__(self, input_size, H, output_size):
#         super(NewModel, self).__init__()
#         self.linear = torch.nn.Linear(input_size, H, bias=True)
#         self.linear2 = torch.nn.Linear(H, H, bias=True)
#         self.linear3 = torch.nn.Linear(H, H, bias=True)
#         self.linear4 = torch.nn.Linear(H, output_size)
#
#     def forward(self, x):
#         x = torch.tanh(self.linear(x))
#         x = torch.tanh(self.linear2(x))
#         x = torch.tanh(self.linear3(x))
#         x = self.linear4(x)
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
def train_local_model(l_model, train_data, local_epoch, initial_learning_rate):
    # Define your loss function and optimizer
    # criterion = nn.NLLLoss()  # 这个损失函数不太好
    criterion = nn.CrossEntropyLoss()
    # 创建优化器时，将学习率设为一个变量
    # optimizer = optim.SGD(model.parameters(), lr=0.005)
    # optimizer = optim.Adam(model.parameters())  # 这个优化器不太好
    optimizer = optim.SGD(l_model.parameters(), lr=initial_learning_rate)
    for epoch in range(local_epoch):
        inputs, labels = train_data
        l_model.train()
        labels = labels.long()
        optimizer.zero_grad()
        inputs = inputs.float()
        outputs = l_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# def train_local_model(l_model, train_data, local_epoch, initial_learning_rate):
#     mu = 0.1
#     optimizer = torch.optim.SGD(l_model.parameters(), lr=initial_learning_rate)  # initial_learning_rate还可以=0.01
#     criterion = nn.CrossEntropyLoss()
#     global_weights = copy.deepcopy(list(l_model.parameters()))
#     for epoch in range(local_epoch):
#         l_model.train()
#         inputs, labels = train_data
#         labels = labels.long()
#         inputs = inputs.float()
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         # FedProx
#         # prox_term = 0.0
#         # for p_i, param in enumerate(model.parameters()):
#         #         prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
#         # loss += prox_term
#
#         loss.backward()
#         optimizer.step()


# Calculate accuracy
def calculate_accuracy(model, test_data):
    correct = 0

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        inputs, labels = test_data
        inputs = inputs.float()

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
        # csv_name = f"data/realdatasets/{csv_name}_{i}.csv"
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


def driftDetection(window_data):
    """使用ks_2samp方法分别测试数据每个维度,使用前一段和后一段的数据"""
    d_points = []
    p = 0
    for d_point in window_data.get_window_data():
        d_points.append(d_point[0].cpu().numpy())
    # print(f"window_data[]={window_data[0]}")
    num_features = d_points[0].shape[1]  # 使用第一个数组来获取特征数量
    feature_data = [[] for _ in range(num_features)]
    # print("d_points 中有", num_features, "个特征。")
    # 提取每个特征的数据
    for n in range(num_features):
        feature_data[n] = np.concatenate([data[:, n] for data in d_points], axis=0)
        k = len(feature_data[n])
        batch1 = feature_data[n][: int(k / 2)]
        batch2 = feature_data[n][int(k / 2) + 1: int(k)]
        # 使用 ks_2samp检测均值是否发生显著变化
        ks_statistic, p_value = stats.ks_2samp(batch1, batch2)
        p += p_value

    # print(f"p={p}")
    if p < alpha:
        # print("发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
        return True
    else:
        # print("未发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
        return False

    # """使用ks_2samp方法测试数据总的"""
    # d_points = []
    #
    # for d_point in window_data.get_window_data():
    #     d_points.append(d_point[0].cpu().numpy())
    # # print(f"window_data[]={window_data[0]}")
    # # print("d_points=", d_points)
    # feature_data = np.concatenate(d_points, axis=0)
    # # print("feature_data")
    # k = len(feature_data) / 2
    # N = len(feature_data)
    # # print(f"k={k},N={N}")
    # batch1 = feature_data[: int(k)]
    # batch2 = feature_data[int(k) + 1: int(N)]
    # scalars_np_1 = batch1.ravel()  # 不一定要合并为一维，各个维度都可以检测一下，试试多维检测法，试一下第0维，试试聚类，更改数据集
    # scalars_np_2 = batch2.ravel()
    # # 使用 t-检验检测均值是否发生显著变化
    # ks_statistic, p_value = stats.ks_2samp(scalars_np_1, scalars_np_2)
    #
    # if p_value < alpha:
    #     # print("发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
    #     return True
    # else:
    #     # print("未发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
    #     return False


def estimateParams(data, padding):
    mean = np.mean(data)
    std = np.std(data)
    alpha = ((1 - padding) * mean / std ** 2 - 1) * mean
    beta = alpha * (1 / mean - 1)
    return alpha, beta


def z_score_scaling(data):  # 将特征值缩放为均值为 0，标准差为 1 的标准正态分布。
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    scaled_data = (data - mean) / std_dev
    return scaled_data


def normal_pdf(x, mean, std_dev):
    """
    正态分布的概率密度函数 (PDF)
    """
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(exponent)
    print(f"在均值为 {mean}，标准差为 {std_dev} 的正态分布中，x = {x} 的概率密度为 {pdf:.4f}")
    return pdf


def calculateLogLikelihood(x, alpha, beta):
    return gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) + (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x)


def federated_stream_artificial():
    # 创建一个 FixedSizeWindow 对象列表，每个客户端一个，最大容纳 10 段数据流
    window_data_list = [FixedSizeWindow(4) for _ in range(num_clients)]
    drift_points = []  # 用于存储漂移点的列表
    initial_learning_rate = 0.005
    # 人工数据集
    clients_data, data_streams = upload_data(drift_Flie_name, num_clients, batch_size)

    # global_model = copy.deepcopy(model)
    global_model_drift = copy.deepcopy(model)
    global_model_Notdrift = copy.deepcopy(model)
    local_model = [copy.deepcopy(model) for _ in range(num_clients)]
    global_test_acc = []
    client_accuracy = [[] for _ in range(num_clients)]
    client_change = {}
    for epoch in range(global_epochs):
        total_accuracy = 0.0
        # local_models = []
        local_models_drift = []
        local_models_Notdrift = []
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
            client_accuracy[client_id].append(accuracy)
            # print(f"第{epoch}个全局模型对client_id", client_id, "的测试准确率= ", accuracy)
            total_accuracy += accuracy

            train_local_model(local_model[client_id], client_data, local_epoch, initial_learning_rate)

            # 检测漂移
            ifdetection = driftDetection(window_data_list[client_id])
            if ifdetection:
                # print(f"在epoch{epoch}客户端{client_id}数据流发生漂移！")
                drift_points.append((epoch, client_id))  # 记录漂移点
                client_change[client_id] = 1

            if client_change.get(client_id) == 1:
                local_models_drift.append(local_model[client_id])
            else:
                local_models_Notdrift.append(local_model[client_id])
            # local_models.append(local_model[client_id])

        # # Average local models into the global model
        # 将本地模型的参数平均到全局模型中
        for global_drift_param, local_param_list in zip(global_model_drift.parameters(),
                                                   zip(*[local_model[client_id].parameters() for
                                                        local_model[client_id] in local_models_drift])):
            global_drift_param.data = torch.mean(torch.stack(local_param_list), dim=0)

        for global_Notdrift_param, local_param_list in zip(global_model_Notdrift.parameters(),
                                                   zip(*[local_model[client_id].parameters() for
                                                        local_model[client_id] in local_models_Notdrift])):
            global_Notdrift_param.data = torch.mean(torch.stack(local_param_list), dim=0)
        # for global_param, local_param_list in zip(global_model.parameters(),
        #                                            zip(*[local_model[client_id].parameters() for
        #                                                 local_model[client_id] in local_models])):
        #     global_param.data = torch.mean(torch.stack(local_param_list), dim=0)

        # 在每个客户端上加载全局模型的参数
        # for client_id, local_model in enumerate(local_models):
        #     # 检查客户端的client_change是否为1
        #     if client_change.get(client_id) == 1:
        #         # 将客户端模型更新为全局模型的状态字典
        #         local_model.load_state_dict(global_model.state_dict())
        # for client_id in local_models:
        #     if client_change.get(client_id) == 1:
        #         local_model[client_id].load_state_dict(global_model.state_dict())
        for local_model[client_id] in local_models_drift:
            print(f"drift_client_id={client_id}")
            local_model[client_id].load_state_dict(global_model_drift.state_dict())
        for local_model[client_id] in local_models_Notdrift:
            print(f"Notdrift_client_id={client_id}")
            local_model[client_id].load_state_dict(global_model_Notdrift.state_dict())

        # 全局准确率是所有客户端准确率的平均值，从epoch1开始是第1个全局模型参数上传给每个客户端后测试第2段数据流的准确率
        global_accuracy = total_accuracy / len(clients_data)

        # print(f"----------------------Epoch {epoch}: Global Model Accuracy = {global_accuracy}")
        global_test_acc.append(global_accuracy)

    # 画客户正确率图、
    # for i in range(num_clients):
    plt.figure(figsize=(10, 5))
    client_accuracys = []
    if len(client_accuracys) == 0:
        client_accuracys = client_accuracy
    else:
        client_accuracys = np.array(client_accuracys)
        client_accuracy = np.array(client_accuracy)
        client_accuracys = client_accuracys + client_accuracy
        client_accuracys = client_accuracys.tolist()

    for i, avg_acc in enumerate(client_accuracys):
        plt.plot(avg_acc, label=f"Client {i + 1}")
    plt.xlabel('Time stamp')
    plt.ylabel('Test accs')
    plt.title('Test accs of Clients')
    plt.ylim(0, 101)
    plt.xlim(0, global_epochs)
    plt.legend()
    plt.savefig(f"./new-results/client_ks2samp_ED_alp{alpha}_{test_name}_acc.png")
    plt.show()

    # 画全局正确率图
    # global_test_accs = []
    # if len(global_test_accs) == 0:
    #     global_test_accs = global_test_acc
    # else:
    #     global_test_accs = np.array(global_test_accs)
    #     global_test_acc = np.array(global_test_acc)
    #     global_test_accs = global_test_accs + global_test_acc
    #     global_test_accs = global_test_accs.tolist()

    # 设置Matplotlib后端
    # plt.switch_backend('TkAgg')  # 使用TkAgg后端
    # 创建Figure
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(global_test_acc)), global_test_acc)
    # plt.xlabel('Time stamp')
    # plt.ylabel('Test Accuracy')
    # plt.ylim(0, 101)
    # plt.title('Global Test Accuracy')
    # plt.grid(True)
    # plt.savefig(f"./new-results/global_ks2samp_ED_alp{alpha}_{test_name}_acc.png")
    #

    # 绘制漂移点图
    plt.figure(figsize=(10, 5))

    if drift_points:
        epochs, client_ids = zip(*drift_points)
        plt.scatter(epochs, client_ids, c='red', marker='o', label='Drift Points')

    plt.xlabel('Epoch')
    plt.ylabel('Client ID')
    plt.title('Drift Points')

    # 设置 x 轴的范围为所有的 epoch
    plt.xlim(0, global_epochs)

    # 保存漂移点图
    plt.savefig(f"./new-results/driftp_ks2samp_ED_alp{alpha}_{test_name}.png")
    plt.show()
    plt.pause(0.1)

    # Return global test accuracy and drift points list
    return global_test_acc


if __name__ == "__main__":
    global_epochs = 200
    local_epoch = 1
    num_clients = 10
    batch_size = 100
    gpu = 0
    # sensitivity = 0.1
    Nmax = 10
    padding = 50
    # 设置检测显著性水平
    alpha = 0.01

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    print(device)
    model = NewModel(input_size=2, hidden_size=512, output_size=2)  # hidden_size
    model = model.to(device)
    test_name = f"generated_hyperplane_data_2D_0.csv_g{global_epochs}_l{local_epoch}"
    drift_Flie_name = 'generated_hyperplane_data_2D'  # generated_hyperplane_data_2D_0.csv

    num_iterations = 1
    global_accuracy_sum = [0.0] * global_epochs

    for i in range(num_iterations):
        print(f"Iteration {i + 1}/{num_iterations}")
        global_accuracy = federated_stream_artificial()  # 注意修改这里，只接收全局准确率
        # global_accuracy = federated_stream_real()
        for j, acc in enumerate(global_accuracy):
            global_accuracy_sum[j] += acc  # 将每次迭代的全局准确率累加到全局准确率列表中

    average_global_accuracy = [acc / num_iterations for acc in global_accuracy_sum]  # 计算平均值

    # 绘制图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(global_epochs), average_global_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Average Test Accuracy')
    plt.ylim(0, 101)
    plt.xlim(0, global_epochs)
    plt.title('Average Global Test Accuracy over 30 Iterations')
    plt.grid(True)
    plt.savefig(f"./new-results/{drift_Flie_name}_average_global_accuracy.png")
    plt.show()
    # federated_stream_artificial()
    # federated_stream_real()
