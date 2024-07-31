import numpy as np
import pandas as pd

datasets = []
# 定义概念a, b, c的数据生成函数
def concept_a():
    # 这里可以定义概念a的数据生成逻辑，例如均值、标准差等
    # 假设概念a的数据分布是均值为0，标准差为1的正态分布
    return np.random.normal(0, 1, (20000, 8))


def concept_b():
    # 这里可以定义概念b的数据生成逻辑，例如均值、标准差等
    # 假设概念b的数据分布是均值为1，标准差为1的正态分布
    return np.random.normal(1, 1, (20000, 8))


def concept_c():
    # 这里可以定义概念c的数据生成逻辑，例如均值、标准差等
    # 假设概念c的数据分布是均值为-1，标准差为1的正态分布
    return np.random.normal(-1, 1, (20000, 8))


# 创建数据集
def create_dataset(concept):
    # 生成概念a的数据
    data = concept()

    # 应用一个随机的变换矩阵，模拟数据分布的变化
    transformation_matrix = np.random.randn(8, 8)
    transformed_data = np.dot(data, transformation_matrix) + np.random.randn(1, 8)

    # 打乱数据点
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data[shuffled_indices]

    # 按规定顺序组合 A 和 B/C 类数据
    if concept == concept_b:
        concatenated_data = np.concatenate([
            shuffled_data[:10000],  # 概念a
            shuffled_data[10000:]  # 概念b
        ])
    elif concept == concept_c:
        concatenated_data = np.concatenate([
            shuffled_data[:10000],  # 概念a
            shuffled_data[10000:]  # 概念c
        ])

    # 创建DataFrame
    df = pd.DataFrame(concatenated_data, columns=[f'Feature_{i}' for i in range(1, 9)])
    df['Label'] = np.hstack((np.zeros(10000), np.ones(10000)))

    return df


# 创建10个数据集
for i in range(2):
    datasets.append(create_dataset(concept_b))

for i in range(8):
    datasets.append(create_dataset(concept_c))

# 保存数据集到CSV文件
for i, df in enumerate(datasets):
    df.to_csv(f'direction_half_{i}.csv', index=False)

