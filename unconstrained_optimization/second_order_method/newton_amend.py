import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encapsulation.Golden_cut import golden_cut
from encapsulation.Plot import plot
from encapsulation.Stop_condition import stop_condition
from encapsulation.test_function import f,x_0

import numpy as np
from autograd import grad,hessian

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

x_list, y_list = newton(f, x_0, 1e-3)
print(x_list,y_list)
plot(x_list, y_list, 2)