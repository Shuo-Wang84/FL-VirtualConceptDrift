import numpy as np
import pandas as pd

# 生成具有更高复发频率的概念漂移数据集
for i in range(10):
    np.random.seed(i * 11)
    n_samples = 50000  # 总共20000个数据

    # 生成两类数据，一类用红色，一类用蓝色
    red_data = np.random.randn(n_samples // 2, 8) * 0.5 + np.array([2, 2, 2, 2, 2, 2, 2, 2])
    blue_data = np.random.randn(n_samples // 2, 8) * 0.5 + np.array([-2, -2, -2, -2, -2, -2, -2, -2])

    # 创建一个数据集，将红色和蓝色的数据合并在一起，并创建相应的标签
    data_V = np.vstack((red_data, blue_data))
    labels = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

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

    # 按规定顺序组合 A 和 B 类数据，实现更高的复发频率
    segment_size = n_samples // 20
    concatenated_data = np.concatenate([
        shuffled_data[:segment_size],  # 原始数据
        shuffled_transformed_data[:segment_size],  # 变换后的数据
        shuffled_data[segment_size:2 * segment_size],  # 原始数据
        shuffled_transformed_data[segment_size:2 * segment_size],  # 变换后的数据
        shuffled_data[2 * segment_size:3 * segment_size],  # 原始数据
        shuffled_transformed_data[2 * segment_size:3 * segment_size],  # 变换后的数据
        shuffled_data[3 * segment_size:]  # 原始数据
    ])
    concatenated_labels = np.concatenate([
        shuffled_labels[:segment_size],
        shuffled_transformed_labels[:segment_size],
        shuffled_labels[segment_size:2 * segment_size],
        shuffled_transformed_labels[segment_size:2 * segment_size],
        shuffled_labels[2 * segment_size:3 * segment_size],
        shuffled_transformed_labels[2 * segment_size:3 * segment_size],
        shuffled_labels[3 * segment_size:]
    ])

    # 保存拼接的数据到CSV文件
    df = pd.DataFrame(np.column_stack((concatenated_data, concatenated_labels)),
                      columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6',
                               'Feature 7', 'Feature 8', 'Label'])

    df.to_csv(f'MD_X-hight_recurrence_data_{i}.csv', index=False)
