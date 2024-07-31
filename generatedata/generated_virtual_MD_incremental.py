import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 生成原始数据
for i in range(10):
    # 创建一个空的 DataFrame，用于存储所有的逐渐漂移后的数据
    df_combined = pd.DataFrame()

    np.random.seed(i * 11)
    n_samples = 2000   # 总共20000个数据，数据分布改变发生在10000

    # 生成两类数据，一类用红色，一类用蓝色
    red_data = np.random.randn(n_samples, 8) * 0.5 + np.array([2, 2, 2, 2, 2, 2, 2, 2])
    blue_data = np.random.randn(n_samples, 8) * 0.5 + np.array([-2, -2, -2, -2, -2, -2, -2, -2])

    # 创建一个数据集，将红色和蓝色的数据合并在一起，并创建相应的标签
    data_V = np.vstack((red_data, blue_data))
    labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    # 对数据进行逐渐变换
    transformation_matrix = np.eye(8)  # 初始为单位矩阵
    offset_vector = np.zeros(8)  # 初始偏移向量为0

    transformed_data_list = []

    for j in range(10):
        transformation_matrix = 0.9 * transformation_matrix + 0.1 * np.random.randn(8, 8)  # 逐渐改变变换矩阵
        offset_vector = 0.9 * offset_vector + 0.1 * np.random.randn(8)  # 逐渐改变偏移向量

        transformed_data = np.dot(data_V, transformation_matrix) + offset_vector
        transformed_data_list.append(transformed_data)

    # 将逐渐漂移后的数据添加到 DataFrame 中
    for transformed_data in transformed_data_list:
        df_temp = pd.DataFrame(np.column_stack((transformed_data, labels)),
                               columns=['Feature ' + str(k) for k in range(1, 9)] + ['Label'])
        df_temp['Label'] = df_temp['Label'].astype(int)
        df_combined = pd.concat([df_combined, df_temp], ignore_index=True)

    # 将合并后的 DataFrame 保存为 CSV 文件
    df_combined.to_csv(f'gradual_drift_data_{i}.csv', index=False)


