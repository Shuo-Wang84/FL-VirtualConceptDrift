import numpy as np
import matplotlib.pyplot as plt

# 生成原始数据
import pandas as pd

for i in range(10):
    np.random.seed(i * 11)
    n_samples = 5000  # 总共20000个数据，数据分布改变发生在10000
    # n_samples = 5000+i*100   #异构

    # 生成两类数据，一类用红色，一类用蓝色
    red_data = np.random.randn(n_samples, 8) * 0.5 + np.array([2, 2, 2, 2, 2, 2, 2, 2])
    blue_data = np.random.randn(n_samples, 8) * 0.5 + np.array([-2, -2, -2, -2, -2, -2, -2, -2])
    # red_data = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    # blue_data = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, -2])

    # 创建一个数据集，将红色和蓝色的数据合并在一起，并创建相应的标签
    data_V = np.vstack((red_data, blue_data))
    labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    # 绘制原始数据
    plt.scatter(data_V[labels == 0, 0], data_V[labels == 0, 1], c='red', label='Class 0', alpha=0.5)
    plt.scatter(data_V[labels == 1, 0], data_V[labels == 1, 1], c='blue', label='Class 1', alpha=0.5)

    # 对数据进行一些变换，例如旋转和平移
    # if i>=5:
    # transformation_matrix = np.array([[4, -0.1], [4, 1.5]])
    # transformed_data = np.dot(data_V, transformation_matrix) + np.array([2, -1])
    # transformed_labels = labels
    # transformation_matrix = np.array([[10, -2], [10, -1.5]])
    # transformed_data = np.dot(data_V, transformation_matrix) + np.array([4, -2])
    # transformed_labels = labels

    # 这里是多维
    # transformation_matrix = np.random.randn(8, 8)
    # transformed_data = np.dot(data_V, transformation_matrix) + np.random.randn(1, 8)
    # transformed_labels = labels

    # 多维增加变化严重性
    transformation_matrix = np.random.randn(8, 8) * 10  # 放大变换矩阵的影响，乘以一个较大的系数
    # random_noise = np.random.randn(len(data_V), 8)   # 调整随机扰动的标准差，增加随机性
    transformed_data = np.dot(data_V, transformation_matrix) + np.random.randn(1, 8)  # 添加随机扰动
    transformed_labels = labels  # 标签保持不变

    # else:
    # red_data_t = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    # blue_data_t = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, -2])
    # red_data_t=np.random.randn(n_samples, 8) * 0.5 + np.array([2, 2, 2, 2, 2, 2, 2, 2])
    # blue_data_t=np.random.randn(n_samples, 8) * 0.5 + np.array([-2, -2, -2, -2, -2, -2, -2, -2])
    # transformed_data = np.vstack((red_data_t, blue_data_t))
    # transformed_labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    # 绘制变换后的数据
    plt.scatter(transformed_data[labels == 0, 0], transformed_data[labels == 0, 1], c='red', marker='x',
                label='Class 0 (Transformed)')
    plt.scatter(transformed_data[labels == 1, 0], transformed_data[labels == 1, 1], c='blue', marker='x',
                label='Class 1 (Transformed)')

    # plt.legend()
    # plt.title('Data Distribution with Label Change')
    # plt.show()

    # 打乱数据点
    shuffled_indices = np.random.permutation(len(data_V))
    shuffled_data = data_V[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    shuffled_transformed_indices = np.random.permutation(len(transformed_data))
    shuffled_transformed_data = transformed_data[shuffled_transformed_indices]
    shuffled_transformed_labels = transformed_labels[shuffled_transformed_indices]

    # 垂直拼接数据
    concatenated_data = np.vstack((shuffled_data, shuffled_transformed_data))
    concatenated_labels = np.hstack((shuffled_labels, shuffled_transformed_labels))

    # 保存拼接的数据到CSV文件
    df = pd.DataFrame(np.column_stack((concatenated_data, concatenated_labels)),
                      columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6',
                               'Feature 7', 'Feature 8', 'Label'])
    # df = pd.DataFrame(np.column_stack((concatenated_data, concatenated_labels)),
    #                   columns=['Feature 1', 'Feature 2', 'Label'])

    df.to_csv(f'vitual_sudden_MD_Syn_all_L_{i}.csv', index=False)
    # df.to_csv(f'vitual_sudden_Syn_all_H_{i}.csv', index=False)

    # ... 在生成和变换数据的代码之后

    # # 假设有一个线性分类器生成的分界线参数，如斜率和截距
    # slope = -0.5
    # intercept = 0
    #
    # # 生成 x 值范围
    # x_values = np.linspace(min(concatenated_data[:, 0]), max(concatenated_data[:, 0]), 100)
    #
    # # 根据线性方程计算 y 值
    # y_values = slope * x_values + intercept

    # 绘制红色的分界线
    # plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')
    plt.legend()
    plt.title(f'gussian_MD_Syn_sll_{i}')

    # 保存图像
    plt.savefig(f'gussian_MD_Syn_sll_{i}.png')
    plt.show()
