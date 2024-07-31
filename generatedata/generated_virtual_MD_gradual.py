import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 生成原始数据
for i in range(10):
    np.random.seed(i * 11)
    n_samples = 9000  # 总共20000个数据，数据分布改变发生在10000

    # 生成两类数据，一类用红色，一类用蓝色
    red_data = np.random.randn(n_samples, 8) * 0.5 + np.array([2, 2, 2, 2, 2, 2, 2, 2])
    blue_data = np.random.randn(n_samples, 8) * 0.5 + np.array([-2, -2, -2, -2, -2, -2, -2, -2])

    # 创建一个数据集，将红色和蓝色的数据合并在一起，并创建相应的标签
    data_V = np.vstack((red_data, blue_data))
    labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    # 对数据进行一些变换，例如旋转和平移
    transformation_matrix = np.random.randn(8, 8)
    transformed_data = np.dot(data_V, transformation_matrix) + np.random.randn(1, 8)
    transformed_labels = labels

    # 打乱数据点
    shuffled_indices = np.random.permutation(len(data_V))
    shuffled_data = data_V[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    shuffled_transformed_indices = np.random.permutation(len(transformed_data))
    shuffled_transformed_data = transformed_data[shuffled_transformed_indices]
    shuffled_transformed_labels = transformed_labels[shuffled_transformed_indices]

    # 按规定顺序组合 A 和 B 类数据
    concatenated_data = np.concatenate([
        shuffled_data[:2000],
        shuffled_transformed_data[:1000],
        shuffled_data[2000:3000],
        shuffled_transformed_data[1000:4000],
        shuffled_data[3000:4000],
        shuffled_transformed_data[4000:]
    ])
    concatenated_labels = np.concatenate([
        shuffled_labels[:2000],
        shuffled_transformed_labels[:1000],
        shuffled_labels[2000:3000],
        shuffled_transformed_labels[1000:4000],
        shuffled_labels[3000:4000],
        shuffled_transformed_labels[4000:]
    ])

    # 保存拼接的数据到CSV文件
    df = pd.DataFrame(np.column_stack((concatenated_data, concatenated_labels)),
                      columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6',
                               'Feature 7', 'Feature 8', 'Label'])

    df.to_csv(f'MD_gradual_data_{i}.csv', index=False)
