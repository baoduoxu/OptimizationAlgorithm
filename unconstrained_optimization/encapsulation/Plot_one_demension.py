import numpy as np
import matplotlib.pyplot as plt

def plot_1(f,left=0 ,right=5):
    x = np.linspace(left, right, 1000)  # 定义自变量x的范围和数量
    y = [f(x[i])for i in range(1000)]                     # 计算函数的取值
    # print(y)
    plt.plot(x, y)              # 绘制函数图像
    plt.xlabel('x')             # 设置x轴标签
    plt.ylabel('y')             # 设置y轴标签
    plt.title('Function Plot')  # 设置图像标题
    plt.show()                  # 显示图像
