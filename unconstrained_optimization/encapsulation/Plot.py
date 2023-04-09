from encapsulation.test_function import f
import matplotlib.pyplot as plt
import numpy as np
def plot(x_list, y_list, graph_type=2):
    x = np.linspace(-12, 12, 1000)  # 绘图区间
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)  # 构造网格
    Z = f([X, Y])
    if graph_type == 2:  # 二维图
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.plot([x[0] for x in x_list], [x[1]
                 for x in x_list], marker='.', color='r', linewidth=0.5)
        # 设置坐标轴标签
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    elif graph_type == 3:  # 三维图
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.plot([x[0] for x in x_list], [x[1]
                for x in x_list], y_list, marker='.', color='r', linewidth=0.5)
        # 设置坐标轴标签
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()