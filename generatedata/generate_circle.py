import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for i in range(1):
    if i >= 0: #生成half>5就是漂移，小于5不漂移
        """1、4个圆圈"""
        # 生成数据 Asyn_epoch=100，102，104，106，108，110，112，114，116，118，120
        num_points = 2500  # + i * 50  # 2500*4=10000 也就是100epoch
        theta = np.linspace(0, 2 * np.pi, num_points)

        # 类别A的数据
        radius_a = 5 + i
        center_a = (10 + i, 10 + i)
        x_a = center_a[0] + radius_a * np.cos(theta)
        y_a = center_a[1] + radius_a * np.sin(theta)

        # 类别B的数据
        radius_b = 5 + i
        center_b = (-10 - i, 10 + i)
        x_b = center_b[0] + radius_b * np.cos(theta)
        y_b = center_b[1] + radius_b * np.sin(theta)

        # 类别C的数据
        radius_c = 5 + i
        center_c = (-10 - i, -10 - i)
        x_c = center_c[0] + radius_c * np.cos(theta)
        y_c = center_c[1] + radius_c * np.sin(theta)

        # 类别D的数据
        radius_d = 5 + i
        center_d = (10 + i, -10 - i)
        x_d = center_d[0] + radius_d * np.cos(theta)
        y_d = center_d[1] + radius_d * np.sin(theta)

        # 创建DataFrame
        df_A = pd.DataFrame({'X': x_a, 'Y': y_a, 'Label': '0'})
        df_B = pd.DataFrame({'X': x_b, 'Y': y_b, 'Label': '1'})
        df_C = pd.DataFrame({'X': x_c, 'Y': y_c, 'Label': '2'})
        df_D = pd.DataFrame({'X': x_d, 'Y': y_d, 'Label': '3'})

        # 合并数据
        df = pd.concat([df_A, df_B, df_C, df_D], ignore_index=True)

        # 随机打乱数据框的顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 将数据保存到CSV文件
        df.to_csv('random_data1.csv', index=False)

        # 绘制图表
        # 生成 x 值范围
        # x_values = np.linspace(0, 1, 100)
        x_values=np.linspace(0, 0, 100)
        # 根据线性方程计算 y 值
        y_values = np.linspace(-15, 15, 100)
        # y_values = slope * x_values + intercept
        x_values2 = np.linspace(-15, 15, 100)
        y_values2 = np.linspace(0, 0, 100)

        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(8, 8))
        plt.scatter(x_a, y_a, label='Class A')
        plt.scatter(x_b, y_b, label='Class B')
        plt.scatter(x_c, y_c, label='Class C')
        plt.scatter(x_d, y_d, label='Class D')
        plt.rcParams.update({'font.size': 16})
        # 绘制红色的分界线
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')
        plt.plot(x_values2, y_values2, color='red', linestyle='--', label='Decision Boundary')
        plt.legend()
        plt.title(f'circle_concept1_{i}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        # 保存图像
        plt.savefig(f'circle_concept1_{i}.png')
        plt.show()

        """2、4个扇形"""
        # 设置随机种子，以确保结果可复现
        np.random.seed(42)

        # 生成A、B、C、D四个象限的数据 Asyn_epoch=300，298，296，294，292，290，288，286，284，282，280
        num_samples1 = 5000  # -i*100  # 5000*4=20000
        radius_3 = np.random.uniform(0 + i, 10 + i, num_samples1)

        # A象限数据
        theta_a = np.random.uniform(0, np.pi / 2, num_samples1)
        xa = 0 + radius_3 * np.cos(theta_a)
        ya = 0 + radius_3 * np.sin(theta_a)

        # B象限数据
        theta_b = np.random.uniform(np.pi / 2, np.pi, num_samples1)
        xb = 0 + radius_3 * np.cos(theta_b)
        yb = 0 + radius_3 * np.sin(theta_b)

        # C象限数据
        theta_c = np.random.uniform(np.pi, 3 * np.pi / 2, num_samples1)
        xc = 0 + radius_3 * np.cos(theta_c)
        yc = 0 + radius_3 * np.sin(theta_c)

        # D象限数据
        theta_d = np.random.uniform(3 * np.pi / 2, 2 * np.pi, num_samples1)
        xd = 0 + radius_3 * np.cos(theta_d)
        yd = 0 + radius_3 * np.sin(theta_d)

        # 创建DataFrame
        df_A = pd.DataFrame({'X': xa, 'Y': ya, 'Label': '0'})
        df_B = pd.DataFrame({'X': xb, 'Y': yb, 'Label': '1'})
        df_C = pd.DataFrame({'X': xc, 'Y': yc, 'Label': '2'})
        df_D = pd.DataFrame({'X': xd, 'Y': yd, 'Label': '3'})

        # 合并数据
        df = pd.concat([df_A, df_B, df_C, df_D], ignore_index=True)

        # 随机打乱数据框的顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 将数据保存到CSV文件
        df.to_csv('random_data2.csv', index=False)

        x_values = np.linspace(0, 0, 100)
        # 根据线性方程计算 y 值
        y_values = np.linspace(-10, 10, 100)
        # y_values = slope * x_values + intercept
        x_values2 = np.linspace(-10, 10, 100)
        y_values2 = np.linspace(0, 0, 100)

        plt.rcParams.update({'font.size': 15})

        plt.figure(figsize=(8, 8))
        # 绘制散点图
        plt.scatter(xa, ya, label='Class A')
        plt.scatter(xb, yb, label='Class B')
        plt.scatter(xc, yc, label='Class C')
        plt.scatter(xd, yd, label='Class D')

        # 绘制红色的分界线
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')
        plt.plot(x_values2, y_values2, color='red', linestyle='--', label='Decision Boundary')
        # 设置图例和标题
        plt.legend()
        plt.title((f'circle_concept2_{i}'))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'circle_concept2_{i}.png')
        # 显示图形
        plt.show()

        """3、4个圆形数据"""
        # 设置随机种子以确保可重复性
        np.random.seed(42)

        # 生成数据  Asyn_epoch=400,402,404,406,408,410,412,414,416,418,420
        num_points1 = 2500  # +i*100
        radius = np.random.uniform(0, 5 + i, num_points1)

        # 类别A
        theta_A = np.random.uniform(0, 2 * np.pi, num_points1)
        x_A = 10 + i + radius * np.cos(theta_A)
        y_A = 10 + i + radius * np.sin(theta_A)

        # 类别B
        theta_B = np.random.uniform(0, 2 * np.pi, num_points1)
        x_B = -10 - i + radius * np.cos(theta_B)
        y_B = 10 + i + radius * np.sin(theta_B)

        # 类别C
        theta_C = np.random.uniform(0, 2 * np.pi, num_points1)
        x_C = -10 - i + radius * np.cos(theta_C)
        y_C = -10 - i + radius * np.sin(theta_C)

        # 类别D
        theta_D = np.random.uniform(0, 2 * np.pi, num_points1)
        x_D = 10 + i + radius * np.cos(theta_D)
        y_D = -10 - i + radius * np.sin(theta_D)

        # 创建DataFrame
        df_A = pd.DataFrame({'X': x_A, 'Y': y_A, 'Label': '0'})
        df_B = pd.DataFrame({'X': x_B, 'Y': y_B, 'Label': '1'})
        df_C = pd.DataFrame({'X': x_C, 'Y': y_C, 'Label': '2'})
        df_D = pd.DataFrame({'X': x_D, 'Y': y_D, 'Label': '3'})

        # 合并数据
        df = pd.concat([df_A, df_B, df_C, df_D], ignore_index=True)

        # 随机打乱数据框的顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 将数据保存到CSV文件
        df.to_csv('random_data3.csv', index=False)

        x_values = np.linspace(0, 0, 100)
        # 根据线性方程计算 y 值
        y_values = np.linspace(-15, 15, 100)
        # y_values = slope * x_values + intercept
        x_values2 = np.linspace(-15, 15, 100)
        y_values2 = np.linspace(0, 0, 100)
        plt.figure(figsize=(8, 8))
        # 绘制散点图
        plt.scatter(x_A, y_A, label='Class A')
        plt.scatter(x_B, y_B, label='Class B')
        plt.scatter(x_C, y_C, label='Class C')
        plt.scatter(x_D, y_D, label='Class D')

        # 绘制红色的分界线
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')
        plt.plot(x_values2, y_values2, color='red', linestyle='--', label='Decision Boundary')
        plt.rcParams.update({'font.size': 16})
        # 设置图表属性
        plt.title((f'circle_concept3_{i}'))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # 保持坐标轴比例一致
        plt.savefig(f'circle_concept3_{i}.png')
        # 显示图表
        plt.show()

        """4、4个扇圈"""
        np.random.seed(42)

        # 生成A、B、C、D四个象限的数据
        num_samples = 5000 #- i * 50

        # A象限数据
        theta_a = np.random.uniform(0, np.pi / 2, num_samples)
        radius_a = 5 + i
        xa_ = 0 + radius_a * np.cos(theta_a)
        ya_ = 0 + radius_a * np.sin(theta_a)

        # B象限数据
        theta_b = np.random.uniform(np.pi / 2, np.pi, num_samples)
        radius_b = 5 + i
        xb_ = 0 + radius_b * np.cos(theta_b)
        yb_ = 0 + radius_b * np.sin(theta_b)

        # C象限数据
        theta_c = np.random.uniform(np.pi, 3 * np.pi / 2, num_samples)
        radius_c = 5 + i
        xc_ = 0 + radius_c * np.cos(theta_c)
        yc_ = 0 + radius_c * np.sin(theta_c)

        # D象限数据
        theta_d = np.random.uniform(3 * np.pi / 2, 2 * np.pi, num_samples)
        radius_d = 5 + i
        xd_ = 0 + radius_d * np.cos(theta_d)
        yd_ = 0 + radius_d * np.sin(theta_d)

        # 创建DataFrame
        df_A = pd.DataFrame({'X': xa_, 'Y': ya_, 'Label': '0'})
        df_B = pd.DataFrame({'X': xb_, 'Y': yb_, 'Label': '1'})
        df_C = pd.DataFrame({'X': xc_, 'Y': yc_, 'Label': '2'})
        df_D = pd.DataFrame({'X': xd_, 'Y': yd_, 'Label': '3'})

        # 合并数据
        df = pd.concat([df_A, df_B, df_C, df_D], ignore_index=True)

        # 随机打乱数据框的顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 将数据保存到CSV文件
        df.to_csv('random_data4.csv', index=False)
        plt.figure(figsize=(8, 8))

        x_values = np.linspace(0, 0, 100)
        # 根据线性方程计算 y 值
        y_values = np.linspace(-5, 5, 100)
        # y_values = slope * x_values + intercept
        x_values2 = np.linspace(-5, 5, 100)
        y_values2 = np.linspace(0, 0, 100)
        # 绘制散点图
        plt.scatter(xa_, ya_, label='Class A')
        plt.scatter(xb_, yb_, label='Class B')
        plt.scatter(xc_, yc_, label='Class C')
        plt.scatter(xd_, yd_, label='Class D')
        # 绘制红色的分界线
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')
        plt.plot(x_values2, y_values2, color='red', linestyle='--', label='Decision Boundary')
        plt.rcParams.update({'font.size': 16})
        # 设置图例和标题
        plt.legend()
        plt.title((f'circle_concept4_{i}'))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'circle_concept4_{i}.png')
        # 显示图形
        plt.show()

        # 读取三个CSV文件
        df1 = pd.read_csv('random_data1.csv')
        df2 = pd.read_csv('random_data2.csv')
        df3 = pd.read_csv('random_data3.csv')
        df4 = pd.read_csv('random_data4.csv')
        # 使用pd.concat垂直合并数据框
        df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)

        # 将合并后的数据框保存到新的CSV文件
        df_combined.to_csv(f'virtual_circle_half_{i}.csv', index=False)

    else:
        """4个扇形"""
        # 设置随机种子，以确保结果可复现
        np.random.seed(0)

        # 生成A、B、C、D四个象限的数据
        num_samples1 = 15000
        radius_3 = np.random.uniform(0 + i, 10 + i, num_samples1)

        # A象限数据
        theta_a = np.random.uniform(0, np.pi / 2, num_samples1)
        xa = 0 + radius_3 * np.cos(theta_a)
        ya = 0 + radius_3 * np.sin(theta_a)

        # B象限数据
        theta_b = np.random.uniform(np.pi / 2, np.pi, num_samples1)
        xb = 0 + radius_3 * np.cos(theta_b)
        yb = 0 + radius_3 * np.sin(theta_b)

        # C象限数据
        theta_c = np.random.uniform(np.pi, 3 * np.pi / 2, num_samples1)
        xc = 0 + radius_3 * np.cos(theta_c)
        yc = 0 + radius_3 * np.sin(theta_c)

        # D象限数据
        theta_d = np.random.uniform(3 * np.pi / 2, 2 * np.pi, num_samples1)
        xd = 0 + radius_3 * np.cos(theta_d)
        yd = 0 + radius_3 * np.sin(theta_d)

        # 创建DataFrame
        df_A = pd.DataFrame({'X': xa, 'Y': ya, 'Label': '0'})
        df_B = pd.DataFrame({'X': xb, 'Y': yb, 'Label': '1'})
        df_C = pd.DataFrame({'X': xc, 'Y': yc, 'Label': '2'})
        df_D = pd.DataFrame({'X': xd, 'Y': yd, 'Label': '3'})

        # 合并数据
        df = pd.concat([df_A, df_B, df_C, df_D], ignore_index=True)

        # 随机打乱数据框的顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 将数据保存到CSV文件
        df.to_csv(f'virtual_circle_half_{i}.csv', index=False)
        plt.rcParams.update({'font.size': 16})
        # 绘制散点图
        plt.scatter(xa, ya, label='A')
        plt.scatter(xb, yb, label='B')
        plt.scatter(xc, yc, label='C')
        plt.scatter(xd, yd, label='D')

        # 设置图例和标题
        plt.legend()
        plt.title('Data Distribution in Quadrants')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # 显示图形
        plt.show()
