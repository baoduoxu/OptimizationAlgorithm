import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import line_search
# from scipy.optimize import BFGS
from encapsulation.Golden_cut import golden_cut
from encapsulation.Plot import plot
from encapsulation.Stop_condition import stop_condition
from encapsulation.test_function import f,x_0
from encapsulation.Plot_one_demension import plot_1
import numpy as np
from autograd import grad
def Beta(direction,g,g_last,formula_type):
    # print(g,g_last)
    # print(np.shape(g),np.shape(g_last))
    delta_g=g-g_last
    if formula_type=='HS':
        a=g@delta_g
        b=direction@delta_g
    if formula_type=='PR':
        a=g@delta_g
        b=g_last@g_last
    if formula_type=='FR':
        a=g@g
        b=g_last@g_last
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
    # plot_1(phi,0,2)
    while condition>eps:
        condition_type=1
        search_right=1
        # print(condition)
        print(f(x))
        # if k>10: break
        if k%(n)==0:#每n步初始化方向为当前点的梯度
            direction=-grad_f
            step_best = golden_cut(phi, 0, search_right, 1e-5)
            # step_best=line_search(phi,grad(phi),0.,5.)
            # plot_1(phi,0,2)
            delta=step_best*direction
            # print('step_best=',step_best,delta)
            condition=stop_condition(delta,x,grad_f,condition_type)
            x=x+delta
            x_list.append(x)
            y_list.append(f(x))
        else:
            # plot_1(phi,0,2)
            # step_best = golden_cut(phi, 0, search_right, 1e-5)
            step_best=line_search(phi,grad(phi),0.,5.)
            delta=step_best*direction
            # print('step_best=',step_best,delta)
            condition=stop_condition(delta,x,grad_f,condition_type)
            x=x+delta
            x_list.append(x)
            y_list.append(f(x))
            grad_last=grad_f
            grad_f=grad(f)(x)
            # if np.linalg.norm(grad_f)<eps: break
            beta=Beta(direction, grad_f,grad_last,'FR')
            direction=-grad_f+beta*direction
        k+=1
    print("Total iterations:", k-1)
    return x_list,y_list

x_list,y_list=conjugate_grad(f,x_0,1e-4)
# print(x_list)
# print(y_list)
m=len(y_list)
print(x_list[m-1])
print(y_list[m-1])
plot(x_list,y_list,2)