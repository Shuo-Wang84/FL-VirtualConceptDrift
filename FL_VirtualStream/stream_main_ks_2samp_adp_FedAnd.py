import copy
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


class FedANDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FedANDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def local_update_fedand(client_model, global_model, dual_var, local_data, rho, lr, local_epoch):
    """
    本地更新，基于FedAND伪代码
    xi_k+1 ≈ argmin (fi(xi) + yi_k^T (xi - zk) + ρ/2 ||xi - zk||^2)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(client_model.parameters(), lr=lr)

    for epoch in range(local_epoch):
        inputs, labels = local_data
        inputs, labels = inputs.float(), labels.long()
        client_model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = client_model(inputs)
        global_outputs = global_model(inputs)

        # 局部目标函数：fi(xi) + ρ/2 ||xi - zk||^2
        loss = criterion(outputs, labels) + (rho / 2) * torch.norm(outputs - global_outputs) ** 2
        loss.backward()
        optimizer.step()


def global_update_fedand(global_model, client_models, client_weights):
    """
    服务器端全局更新 zk+1 = sum(p_i * xi_k+1)
    """
    with torch.no_grad():
        for global_param, *local_params in zip(global_model.parameters(),
                                               *[client.parameters() for client in client_models]):
            # 使用加权平均的方法更新全局模型
            weighted_sum = torch.zeros_like(global_param)
            for local_param, weight in zip(local_params, client_weights):
                weighted_sum += weight * local_param.data
            global_param.data = weighted_sum

    return global_model


def update_dual_variable(dual_vars, client_models, global_model, rho):
    """
    更新对偶变量 yi_k+1 = yi_k + ρ (xi_k+1 - zk+1)
    """
    for i in range(len(client_models)):  # 遍历每个客户端
        for dual_var, client_param, global_param in zip(dual_vars[i], client_models[i].parameters(),
                                                        global_model.parameters()):
            # 更新对偶变量 yi_k+1 = yi_k + ρ (xi_k+1 - zk+1)
            dual_var.data = dual_var.data + rho * (client_param.data - global_param.data)

    return dual_vars


def calculate_accuracy_fedand(model, test_data):
    correct = 0

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        inputs, labels = test_data
        inputs = inputs.float().to(device)  # 将 inputs 移动到 GPU 或 CPU
        labels = labels.to(device)  # 将 labels 移动到 GPU 或 CPU

        outputs = model(inputs)
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = (correct / len(inputs)) * 100.0
    return accuracy


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


# Main FedAND process
def federated_stream_fedand():
    # 超参数定义
    global_epochs = 200
    local_epochs = 1
    num_clients = 10
    batch_size = 50
    rho = 0.1  # FedAND 中的超参数
    lr = 0.01  # 学习率
    alpha = 0.03
    drift_Flie_name = 'generated_hyperplane_data_2D'


    # 数据加载与分配
    clients_data, data_streams = upload_data(drift_Flie_name, num_clients, batch_size)

    # 初始化全局模型、客户端模型和对偶变量
    global_model = FedANDModel(input_size=2, hidden_size=256, output_size=2).to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    # 确保 dual_vars 的结构与每个客户端的模型参数结构相同
    dual_vars = [[torch.zeros_like(param) for param in global_model.parameters()] for _ in range(num_clients)]

    # 用于记录测试准确率
    global_test_acc = []

    for epoch in range(global_epochs):
        local_models = []

        for client_id in range(num_clients):
            # 获取客户端数据流
            data_streams[client_id] = iter(data_streams[client_id])
            client_data = next(data_streams[client_id])
            client_data = [d.to(device) for d in client_data]

            # 进行本地模型更新
            local_update_fedand(client_models[client_id], global_model, dual_vars[client_id],
                                client_data, rho, lr, local_epochs)
            local_models.append(client_models[client_id])

        # 全局模型更新
        client_weights = [1 / num_clients] * num_clients  # 简单等权重聚合
        global_model = global_update_fedand(global_model, local_models, client_weights)

        # 更新对偶变量
        dual_vars = update_dual_variable(dual_vars, local_models, global_model, rho)

        # 计算并记录全局模型的测试准确率
        total_accuracy = 0.0
        for client_id in range(num_clients):
            accuracy = calculate_accuracy_fedand(global_model, next(iter(data_streams[client_id])))
            total_accuracy += accuracy
        global_accuracy = total_accuracy / num_clients
        global_test_acc.append(global_accuracy)

        # print(f"Epoch {epoch + 1}: Global Model Accuracy = {global_accuracy:.4f}")

    global_test_accs = []
    if len(global_test_accs) == 0:
        global_test_accs = global_test_acc
    else:
        global_test_accs = np.array(global_test_accs)
        global_test_acc = np.array(global_test_acc)
        global_test_accs = global_test_accs + global_test_acc
        global_test_accs = global_test_accs.tolist()

    # 设置Matplotlib后端
    # plt.switch_backend('TkAgg')  # 使用TkAgg后端
    # 创建Figure
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(global_test_accs)), global_test_accs)
    plt.xlabel('Time stamp')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 101)
    plt.xlim(0, global_epochs)
    # plt.title('Global Test Accuracy')
    plt.grid(True)
    plt.savefig(f"./new-results/fedand_global_ks2samp_ED_alp{alpha}_realdata_acc.png")
    plt.show()

    return global_test_acc


if __name__ == "__main__":
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    global_accuracy = federated_stream_fedand()
