import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encapsulation.Golden_cut import golden_cut
from encapsulation.Plot import plot
from encapsulation.Stop_condition import stop_condition
from encapsulation.test_function import f,x_0

import numpy as np
from autograd import grad

def SteepestDescent(f, x_0, eps=1e-3):
    k = 0  # 迭代次数
    condition = 1  # 结束迭代的条件
    x_list = [x_0]
    y_list = [f(x_0)]
    x = x_0
    while condition > eps:
        grad_f = grad(f)(x)  # 梯度
        def phi(step): return f(x-step*grad_f)
        step_best = golden_cut(phi, 0, 100, 1e-5)
        delta = step_best*grad_f
        condition = stop_condition(delta, x)
        x = x-delta
        x_list.append(x)
        y_list.append(f(x))
        k += 1
    print("Total iterations:", k)
    return x_list, y_list

x_list, y_list = SteepestDescent(f, x_0, 1e-3)
print(x_list,y_list)
plot(x_list, y_list, 2)
