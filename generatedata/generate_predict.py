import numpy as np
import pandas as pd
import random
#异步同质

# 定义概念a, b的数据生成函数
def concept_a():
    # 这里可以定义概念a的数据生成逻辑，例如均值、标准差等
    # 假设概念a的数据分布是均值为0，标准差为1的正态分布
    return np.random.normal(0, 1, (20000, 8))


def concept_b():
    # 这里可以定义概念b的数据生成逻辑，例如均值、标准差等
    # 假设概念b的数据分布是均值为1，标准差为1的正态分布
    return np.random.normal(1, 0.5, (20000, 8))


# 创建数据集
def create_dataset(concept1,concept2):
    # 生成概念a的数据
    data1 = concept1()
    data2 = concept2()

    # 生成随机概念漂移点
    drift_point = random.randint(1000, 20000)
    print(f"drift_point={drift_point}")

    # 应用一个随机的变换矩阵，模拟数据分布的变化
    transformation_matrix = np.random.randn(8, 8)
    transformed_data = np.dot(data1, transformation_matrix) + np.random.randn(1, 8)

    # 按规定顺序组合 A 和 B 类数据
    concatenated_data = np.concatenate([
        data1[:drift_point],  # 概念a
        data2[drift_point:]   # 概念b
    ])

    # 创建DataFrame
    df = pd.DataFrame(concatenated_data, columns=[f'Feature_{i}' for i in range(1, 9)])
    df['Label'] = np.hstack((np.zeros(drift_point), np.ones(20000 - drift_point)))

    return df

datasets = []
# 创建10个数据集
for i in range(10):
    datasets.append(create_dataset(concept_a,concept_b))

# 保存数据集到CSV文件
for i, df in enumerate(datasets):
    df.to_csv(f'MD_predict_{i}.csv', index=False)
