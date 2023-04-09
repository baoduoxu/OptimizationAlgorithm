import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encapsulation.Golden_cut import golden_cut
from encapsulation.Plot import plot
from encapsulation.Stop_condition import stop_condition
from encapsulation.test_function import f,x_0

import numpy as np
from autograd import grad

def Beta(direction,g,g_last,formula_type):
    # print(g,g_last)
    # print(np.shape(g),np.shape(g_last))
    delta_g=g-g_last
    if formula_type=='HS':
        a=g.T@delta_g
        b=direction.T@delta_g
    if formula_type=='PR':
        a=g.T@delta_g
        b=g_last.T@g_last
    if formula_type=='FR':
        a=g.T@g
        b=g_last.T@g_last
    if b<1e-16: b=1e-16
    return a/b

def conjugate_grad(f,x_0,eps):
    k = 1  # 迭代次数
    condition = 1  # 结束迭代的条件
    x_list = [x_0]
    y_list = [f(x_0)]
    x = x_0
    n=len(x)
    grad_f = grad(f)(x)
    direction=-grad_f #初始化方向
    def phi(step): return f(x+step*direction)
    while condition>eps:
        # if k>2: break
        if k%(n)==0:#每n步初始化方向为当前点的梯度
            direction=-grad_f
            step_best = golden_cut(phi, 0, 100, 1e-5)
            delta=step_best*direction
            # print('step_best=',step_best)
            condition=stop_condition(delta,x)
            x=x+delta
            x_list.append(x)
            y_list.append(f(x))
        else:
            step_best = golden_cut(phi, 0, 100, 1e-5)
            delta=step_best*direction
            # print('step_best=',step_best)
            condition=stop_condition(delta,x)
            x=x+delta
            x_list.append(x)
            y_list.append(f(x))
            grad_last=grad_f
            grad_f=grad(f)(x)
            if np.linalg.norm(grad_f)<eps: break
            beta=Beta(direction, grad_f,grad_last,'PR')
            direction=-grad_f+beta*direction
        k+=1
    print("Total iterations:", k-1)
    return x_list,y_list

x_list,y_list=conjugate_grad(f,x_0,1e-7)
# print(x_list)
# print(y_list)
m=len(y_list)
print(x_list[m-1])
print(y_list[m-1])
plot(x_list,y_list,2)