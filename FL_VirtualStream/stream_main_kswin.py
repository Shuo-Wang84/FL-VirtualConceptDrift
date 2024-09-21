import copy
import csv
import math
import time
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
from scipy.stats import ks_2samp


# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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


# Training function for a local model
def train_local_model(model, train_data, local_epoch):
    # Define your loss function and optimizer
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

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
        csv_name = f"{drift_type}"
        csv_name = f"data/vitual/{csv_name}_{i}.csv"
        train_dataset = load_data(csv_name)
        for data_id, datapoint in enumerate(train_dataset):
            clients_data[i].append(datapoint)

    data_streams = []
    for i in range(num_clients):
        data_stream = DataLoader(clients_data[i], batch_size=batch_size, shuffle=False)
        data_streams.append((data_stream))

    return clients_data, data_streams


def driftDetection(window_data):
    # """kswin-ravel"""
    # kswin_detectors = KSWIN(alpha=alpha, window_size=200)  # alpha越大越敏感
    # d_points = []
    # for d_point in window_data.get_window_data():
    #     d_points.append(d_point[0].cpu().numpy())
    #
    # feature_data = np.concatenate(d_points, axis=0)
    # feature_data2 = feature_data.ravel()
    # for i in feature_data2:
    #     # 使用 检测均值是否发生显著变化
    #     kswin_detectors.add_element(i)
    #     if kswin_detectors.detected_change():
    #         # print("发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
    #         return True
    # else:
    #     # print("未发生漂移：KS统计量 = {:.4f}, p-value = {:.4f}".format(ks_statistic, p_value))
    #     return False

    """kswin-ED"""
    d_points = []
    kswin_detectors = KSWIN(alpha=alpha, window_size=50) # alpha越大越敏感
    for d_point in window_data.get_window_data():
        d_points.append(d_point[0].cpu().numpy())
    # print(f"window_data[]={window_data[0]}")
    num_features = d_points[0].shape[1]  # 使用第一个数组来获取特征数量
    feature_data = [[] for _ in range(num_features)]
    # print("d_points 中有", num_features, "个特征。")
    # 提取每个特征的数据
    for n in range(num_features):
        feature_data[n] = np.concatenate([data[:, n] for data in d_points], axis=0)
        # 使用 检测均值是否发生显著变化
        for i in feature_data[n]:
            kswin_detectors.add_element(i)
            if kswin_detectors.detected_change():
                # print('Change detected in data: ' + str(data_point))
                return True
    return False


def federated_stream_artificial():
    # 记录开始时间
    # start_time = time.time()
    # 创建一个 FixedSizeWindow 对象列表，每个客户端一个，最大容纳 10 段数据流
    window_data_list = [FixedSizeWindow(2) for _ in range(num_clients)]

    drift_points = []
    # 人工数据集
    clients_data, data_streams = upload_data(drift_type, num_clients, batch_size)

    global_model = copy.deepcopy(model)
    local_model = [copy.deepcopy(model) for _ in range(num_clients)]
    global_test_acc = []

    for epoch in range(num_epochs):
        total_accuracy = 0.0
        local_models = []  # 不知道这样会不会把local_model[client_id]清除
        confidence_updata = 0
        for client_id in range(num_clients):
            data_streams[client_id] = iter(data_streams[client_id])
            client_data = next(data_streams[client_id])
            client_data = [d.to(device) for d in client_data]

            # 将新的数据流添加到窗口
            window_data_list[client_id].add_data_stream(client_data)

            # Test the global model accuracy on each client's data
            accuracy = calculate_accuracy(local_model[client_id], client_data)
            # print(f"第{epoch}个全局模型对client_id", client_id, "的测试准确率= ", accuracy)
            total_accuracy += accuracy

            # if epoch == 0:
            #     train_local_model(local_model[client_id], client_data, local_epoch)
            # else:
            detection_drift = driftDetection(window_data_list[client_id])
            if detection_drift:
                print(f"在epoch{epoch}客户端{client_id}数据流发生漂移！")
                drift_points.append((epoch, client_id))
                # confidence_updata += 1
                # print("confidence_updata:", confidence_updata)

            train_local_model(local_model[client_id], client_data, local_epoch)
            local_models.append(local_model[client_id])

        # if confidence_updata > 0 or epoch == 0:
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

    # 记录结束时间
    # end_time = time.time()
    #
    # # 计算运行时间
    # elapsed_time = end_time - start_time
    # print("\nElapsed Time: {:.6f} seconds".format(elapsed_time))
    #
    # # 画全局正确率图
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
    # # plt.savefig(f"./new-results/global_kswin_ravel_alp{alpha}_{test_name}_acc.png")

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
    plt.xlim(0, num_epochs)

    # 保存漂移点图
    plt.savefig(f"./new-results/{id}_kswin_alp{alpha}_{drift_type}.png")
    plt.show()
    plt.pause(0.1)


if __name__ == "__main__":
    num_epochs = 200
    local_epoch = 1
    num_clients = 10
    batch_size = 100
    gpu = 0
    sensitivity = 0.1
    Nmax = 10
    padding = 50
    alpha = 0.000000001
    id = 2  # 1：2D 2：Md 3：circle

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    print(device)
    model = MLP(input_size=8, hidden_size=512, output_size=2)
    model = model.to(device)
    # test_name = f"vitual_sudden_g{num_epochs}_l{local_epoch}"
    drift_type = 'vitual_sudden_MD_Asyn'
    federated_stream_artificial()
