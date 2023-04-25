import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encapsulation.Golden_cut import golden_cut
from encapsulation.Plot import plot
from encapsulation.Stop_condition import stop_condition
from encapsulation.test_function import f,x_0

import numpy as np
from autograd import grad

def approx_of_hessian(H,grad_last,grad_f,x_last,x,amend_type):
    delta_x=x_last-x
    # print(grad_last,grad_f)
    delta_g=grad_last-grad_f
    # print(delta_g,delta_x)
    mu1=delta_x.T@delta_g
    mu2=delta_g.T@H@delta_g
    if mu1<1e-16:mu1==1e-16
    if mu2<1e-16:mu2==1e-16
    if amend_type=='rank1':
        tmp=delta_x-H@delta_g
        mu=tmp@delta_g
        if mu<1e-10:mu==1e-10
        return H+(np.outer(tmp,tmp))/mu
    if amend_type=='DFP':
        # print('times: ',delta_x@delta_x,H,H+delta_x@delta_x.T)
        return H+(np.outer(delta_x,delta_x))/mu1-np.outer(H@delta_g,delta_g@H)/mu2
    if amend_type=='BFGS':
        mu3=delta_x@delta_g
        if mu3<1e-16:mu3==1e-16
        # print(mu1,mu2,mu3,np.shape(delta_x),np.shape(delta_g),np.shape(H))
        # print(delta_x@delta_x.T)
        # print(np.shape(np.transpose(delta_g)))
        # print(np.matmul(delta_x,np.transpose(delta_g))) #delta_x@delta_g.T
        # print(delta_x@(delta_g.T)@H)
        # print(H@delta_g@delta_x.T)
        tmp=np.outer(delta_x,delta_g@H)
        return H+((1+mu2/mu1)/mu3)*(np.outer(delta_x,delta_x))-(tmp+tmp.T)/mu3

def quasi_newton(f,x_0,eps):
    k = 0  # 迭代次数
    condition = 1  # 结束迭代的条件
    x_list = [x_0]
    y_list = [f(x_0)]
    x = x_0
    n=len(x)
    H=np.identity(n) #初始化拟牛顿矩阵
    grad_f=grad(f)(x)
    while condition>eps:
        print(condition)
        condition_type=1
        search_right=100
        direction=-H@grad_f
        def phi(step): return f(x+step*direction)
        step_best = golden_cut(phi, 0, search_right, 1e-5)
        # print(step_best)
        delta=step_best*direction
        # print(delta)
        condition=stop_condition(delta,x,grad_f,condition_type)
        x_last=x
        x=x+delta
        x_list.append(x)
        y_list.append(f(x))
        grad_last=grad_f
        grad_f=grad(f)(x)
        H=approx_of_hessian(H,grad_last,grad_f,x_last,x,'DFP' )
        k+=1
    print("Total iterations:", k)
    return x_list,y_list

x_list,y_list=quasi_newton(f,x_0,1e-3)
# print(x_list)
# print(y_list)
m=len(y_list)
print(x_list[m-1])
print(y_list[m-1])
plot(x_list,y_list,2)