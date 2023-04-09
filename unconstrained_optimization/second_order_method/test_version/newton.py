
import math
import numpy as np
from numpy import *
from autograd import grad,hessian
import autograd.numpy
import matplotlib.pyplot as plt

def f(x):  # 在这里编写函数
    return 100*(x[0]-x[1]**2)**2+(1-x[1])**2  # 初始点为[0,0]
    # return x[0]**2+x[1]**2
    # return (x[0]**2)/5+(x[1]**2)/10+autograd.numpy.sin(x[0]+x[1])
x_0 = np.array([0., 0.])  # 在这里定义初始点,请在写浮点数
# print(f(x_0))
rho = (3-math.sqrt(5))/2

def stop_condition(delta, x):
    return np.linalg.norm(delta)/max(1, np.linalg.norm(x))

# def golden_cut(f, l=0, r=100, eps=1e-3):
#     a = l+(r-l)*rho
#     b = r-(r-l)*rho
#     while (math.fabs(l-r) > eps):
#         if f(a) > f(b):
#             l = a
#             a = b
#             b = r-(r-l)*rho
#         else:
#             r = b
#             b = a
#             a = l+(r-l)*rho
#     return a

def plot(x_list, y_list, graph_type=2):
    x = np.linspace(-2, 2, 1000)  # 绘图区间
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)  # 构造网格
    Z = f([X, Y])
    if graph_type == 2:  # 二维图
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.plot([x[0] for x in x_list], [x[1]
                 for x in x_list], marker='.', color='r', linewidth=2)

        # 设置坐标轴标签
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    elif graph_type == 3:  # 三维图
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.plot([x[0] for x in x_list], [x[1]
                for x in x_list], y_list, marker='.', color='r', linewidth=2)
        # 设置坐标轴标签
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

def newton(f, x_0,eps=1e-3):
    # print(f(x_0))
    k = 0  # 迭代次数
    condition = 1  # 结束迭代的条件
    x_list = [x_0]
    y_list = [f(x_0)]
    x = x_0
    while condition > eps:
        # print(k,x,np.shape(f(x)),f(x))
        hessian_f = hessian(f)(x)  # 黑塞矩阵
        grad_f = grad(f)(x)
        # print(x,f(x),grad_f)
        delta=np.linalg.inv(hessian_f)@grad_f
        # print(hessian_f,grad_f)
        condition=stop_condition(delta,x)
        x = x-delta
        x_list.append(x)
        y_list.append(f(x))
        k += 1
    # print("Loccal minimal point:",x_0.evalf(6))
    print("Total iterations:", k)
    return x_list,y_list

x_list, y_list = newton(f, x_0, 1e-3)
print(x_list,y_list)
plot(x_list, y_list, 2)
