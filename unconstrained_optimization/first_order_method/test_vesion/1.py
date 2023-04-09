import numpy as np
import matplotlib.pyplot as plt
import autograd
def gradient_descent(f, initial_x, learning_rate, num_iterations):
    x = initial_x
    x_list = [initial_x]
    y_list = [f(initial_x)]
    for i in range(num_iterations):
        grad = autograd.grad(f)(x)
        x = x - learning_rate * grad
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义函数和初始点
f = lambda x: x[0]**2 + x[1]**2
initial_x = np.array([4.0, 5.0])

# 运行梯度下降算法
x_list, y_list = gradient_descent(f, initial_x, learning_rate=0.1, num_iterations=100)

# 绘制函数曲面
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 绘制最速下降法的路径
ax.plot([x[0] for x in x_list], [x[1] for x in x_list], y_list, marker='o', color='r', linewidth=2)

# 设置坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
