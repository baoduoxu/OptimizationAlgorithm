import math
import numpy as np
from numpy import *
from autograd import grad,hessian
import autograd.numpy
import matplotlib.pyplot as plt
# import newton_high_dim

def f(x):  # 在这里编写函数
    # return 100*(x[0]-x[1]**2)**2+(1-x[1])**2  # 初始点为[0,0]
    # return x[0]**2+x[1]**2
    return (x[0]**2)/5+(x[1]**2)/10+autograd.numpy.sin(x[0]+x[1])
x_0 = np.array([0., 0.])  # 在这里定义初始点,请在写浮点数

def stop_condition(delta, x):
    return np.linalg.norm(delta)/max(1, np.linalg.norm(x))

rho = (3-math.sqrt(5))/2
def golden_cut(f, l=0, r=100, eps=1e-3):
    a = l+(r-l)*rho
    b = r-(r-l)*rho
    while (math.fabs(l-r) > eps):
        if f(a) > f(b):
            l = a
            a = b
            b = r-(r-l)*rho
        else:
            r = b
            b = a
            a = l+(r-l)*rho
    return a

def newton(f, x_0,eps=1e-3):
    k = 0  # 迭代次数
    condition = 1  # 结束迭代的条件
    x_list = [x_0]
    y_list = [f(x_0)]
    x = x_0
    n=len(x)
    while condition > eps:
        hessian_f = hessian(f)(x)  # 黑塞矩阵
        grad_f = grad(f)(x)
        # 求特征值, 如果存在负特征值则给所有的特征值都加上一个mu
        sigma=np.linalg.eigvals(hessian_f)
        m=np.min(sigma)
        mu=0
        if m<0: mu=(-m)+1
        hessian_f+=(mu*np.identity(n))
        direction=np.linalg.inv(hessian_f)@grad_f
        def phi(step): return f(x-step*direction)
        step_best = golden_cut(phi, 0, 100, 1e-5)
        delta=step_best*direction
        condition=stop_condition(delta,x)
        x = x-delta
        x_list.append(x)
        y_list.append(f(x))
        k += 1
    print("Total iterations:", k)
    return x_list,y_list

def plot(x_list, y_list, graph_type=2):
    x = np.linspace(-8, 8, 1000)  # 绘图区间
    y = np.linspace(-8, 8, 1000)
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

x_list, y_list = newton(f, x_0, 1e-3)
print(x_list,y_list)
plot(x_list, y_list, 2)
