import numpy as np
from skmultiflow.data import HyperplaneGenerator
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute the decision boundary line equation
def decision_boundary_line(x):
    return 0.5 * x + 0  # Example equation for a decision boundary (adjust as needed)


for i in range(10):
    if i >= 0:
        # 创建HyperplaneGenerator对象，指定要生成的数据特性和其他参数
        generator1 = HyperplaneGenerator(random_state=0, n_features=2, n_drift_features=2, mag_change=0,
                                         noise_percentage=0,
                                         sigma_percentage=0)  # mag_change=0和sigma_percentage没有任何变化                                sigma_percentage=0) #mag_change=0和sigma_percentage没有任何变化

        # 生成数据
        num_samples = 20000
        data = []
        labels = []
        data_s = []
        data_2 = []
        data_s_2 = []
        labels_2 = []
        for _ in range(num_samples):
            X, y = generator1.next_sample()
            X_s = X.squeeze()
            # 判断数据点位置并分配标签
            if y == 0:
                if X_s[1] > X_s[0]:
                    data.append(X)
                    data_s.append(X_s)
                    labels.append(y)  # 上
                else:

                    data_2.append(X)
                    data_s_2.append(X_s)
                    labels_2.append(y)  #

            if y == 1:
                if X_s[1] < X_s[0]:

                    data.append(X)
                    data_s.append(X_s)
                    labels.append(y)  #
                else:

                    data_2.append(X)
                    data_s_2.append(X_s)
                    labels_2.append(y)  #

        # print(f"data_s={data_s}")
        # print(f"data={data}")
        data = np.array(data)
        labels = np.array(labels)

        # 创建数据帧并保存为CSV文件
        df = pd.DataFrame(data_s, columns=generator1.feature_names)
        df['target'] = labels
        df.to_csv(f'generated_hyperplane_data_2D_1_{i}.csv', index=False)

        # print(f"data_s={data_s}")
        # print(f"data={data}")
        data_2 = np.array(data_2)
        labels_2 = np.array(labels_2)

        # 创建数据帧并保存为CSV文件
        df = pd.DataFrame(data_s_2, columns=generator1.feature_names)
        df['target'] = labels_2
        df.to_csv(f'generated_hyperplane_data_2D_2_{i}.csv', index=False)

        # 读取第一个CSV文件
        df1 = pd.read_csv(f'generated_hyperplane_data_2D_1_{i}.csv')

        # 读取第二个CSV文件
        df2 = pd.read_csv(f'generated_hyperplane_data_2D_2_{i}.csv')

        # 合并两个DataFrame
        merged_df = pd.concat([df1, df2], ignore_index=True)

        # 将合并后的DataFrame写入新的CSV文件
        merged_df.to_csv(f'generated_hyperplane_data_2D_half_{i}.csv', index=False)

        # 读取合并后的CSV文件
        merged_df = pd.read_csv(f'generated_hyperplane_data_2D_half_{i}.csv')

        # 提取特征和标签
        # features = merged_df[['Feature 1', 'Feature 2']].values
        # labels = merged_df['target'].values

        # 假设有一个线性分类器生成的分界线参数，如斜率和截距
        slope = 1
        intercept = 0

        # 生成 x 值范围
        x_values = np.linspace(0, 1, 100)

        # 根据线性方程计算 y 值
        y_values = slope * x_values + intercept

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], label='Class 0', marker='o')
        plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], label='Class 1', marker='x')

        # 绘制红色的分界线
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title(f'Hyperplane Data {i}')
        plt.grid(True)
        # 保存图像
        plt.savefig(f'hyperplane_2D_{i}.png')
        plt.show()

    else:
        # 创建HyperplaneGenerator对象，指定要生成的数据特性和其他参数
        generator1 = HyperplaneGenerator(random_state=i, n_features=2, n_drift_features=2, mag_change=0,
                                         noise_percentage=0,
                                         sigma_percentage=0)  # mag_change=0和sigma_percentage没有任何变化                                sigma_percentage=0) #mag_change=0和sigma_percentage没有任何变化

        # 生成数据
        num_samples = 20000
        data = []
        labels = []
        data_s = []

        for _ in range(num_samples):
            X, y = generator1.next_sample()
            X_s = X.squeeze()
            data.append(X)
            data_s.append(X_s)
            labels.append(y)

        # print(f"data_s={data_s}")
        # print(f"data={data}")
        data = np.array(data)
        labels = np.array(labels)

        # 创建数据帧并保存为CSV文件
        df = pd.DataFrame(data_s, columns=generator1.feature_names)
        df['target'] = labels
        df.to_csv(f'generated_hyperplane_data_2D_half_{i}.csv', index=False)

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], label='Class 0', marker='o')
        plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], label='Class 1', marker='x')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('Scatter Plot of Generated Data')
        plt.grid(True)
        plt.show()
