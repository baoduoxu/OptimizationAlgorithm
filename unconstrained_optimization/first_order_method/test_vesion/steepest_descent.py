# 该文件用来调试, 会有很多注释掉的代码
import math
import numpy as np
from numpy import *
import sympy
from sympy import sympify,lambdify
# import autograd.numpy as np
import autograd
import numexpr as ne
# from TEST import test_function
def f(x):
    return x[0]**2+x[1]**2
x_0=np.array([1.0,1.0])
# def vectorize_func(f_str):
#     var_list = []  # 存储变量名
#     for var in f_str.split():
#         if var.isalpha() and var not in var_list:
#             var_list.append(var)
#     n_var = len(var_list)  # 变量个数
#     print(var_list)
#     def f(*args):
#         if len(args) != n_var:
#             raise ValueError("Number of arguments does not match the number of variables")
#         variables = dict(zip(var_list, args))  # 将变量名和对应的值打包成字典
#         return ne.evaluate(f_str, local_dict=variables)

#     return f
import numexpr as ne
import numpy as np

# def vectorize_func(f_str):
#     var_list = []  # 存储变量名
#     for var in f_str.split():
#         if var.isalpha() and var not in var_list:
#             var_list.append(var)
#     n_var = len(var_list)  # 变量个数
#     print(var_list)
#     def f(*args):
#         if len(args) != n_var:
#             raise ValueError("Number of arguments does not match the number of variables")
#         variables = dict(zip(var_list, args))  # 将变量名和对应的值打包成字典
#         return ne.evaluate(f_str, local_dict=variables)

#     grad_f = []  # 梯度向量的函数
#     for i, var in enumerate(var_list):
#         gradient_str = ne.jacobian(f_str, var)  # 计算偏导数表达式
#         def gradient_func(*args):
#             if len(args) != n_var:
#                 raise ValueError("Number of arguments does not match the number of variables")
#             variables = dict(zip(var_list, args))  # 将变量名和对应的值打包成字典
#             return ne.evaluate(gradient_str, local_dict=variables)
#         grad_f.append(gradient_func)

#     def gradient(*args):
#         if len(args) != n_var:
#             raise ValueError("Number of arguments does not match the number of variables")
#         grad = [grad_f[i](*args) for i in range(n_var)]  # 计算梯度向量的各个分量
#         return np.array(grad)

#     f.gradient = gradient  # 将梯度向量函数作为 f 的属性

#     return f


# def change(var, val):
#     l = []
#     it1 = iter(var)
#     it2 = iter(val)
#     k = var.shape[0]
#     for i in range(k):
#         l.append(tuple((next(it1), next(it2))))
#     return l
rho = (3-math.sqrt(5))/2

def golden_cut(f, l=0, r=100, eps=1e-3):
    # print(f)
    
    # e=float(input()) #精度
    # a=l
    # b=r
    # k=1 #迭代次数
    # print(l,r,e)
    a = l+(r-l)*rho
    b = r-(r-l)*rho
    while (math.fabs(l-r) > eps):
        # print("Iterations:",k)
        # print("a=%.4f" % a)
        # print("b=%.4f" % b)
        # print("f(a)=%.4f" % f(a))
        # print("f(b)=%.4f" % f(b))
        # f_a = f.subs([(var, a)])
        # print(f_a)
        # f_b = f.subs([(var, b)])
        # print(f_b)
        # print(f(a),f(b))
        #print(a,b)
        if f(a) > f(b):
            l = a
            a = b
            b = r-(r-l)*rho
        else:
            r = b
            b = a
            a = l+(r-l)*rho
        # str1="The new section is [{},{}]"
        # print("The new section is","[","%.4f" % l ,",","%.4f" % r,"]")
        # k+=1
    return a


def SteepestDescent(f, x_0, eps=1e-3):
    # grad_f = f.jacobian(var)  # 雅可比矩阵
    k = 0 #迭代次数
    condition = 1 #结束迭代的条件
    x_list=[x_0]
    y_list=[f(x_0)]
    x=x_0
    while condition > eps:
        grad_f=autograd.grad(f)(x) #梯度
        print(grad_f,x)
        # phi=lambda step:f(x-step*grad_f)
        # phi = lambda step: f([x[i] - step * grad_f[i] for i in range(len(x))])
        phi=lambda step:f(x-step*grad_f)
        # print(phi)
        # rel2 = change(var, x0-alpha*d)
        # phi = f.subs(rel2)
        step_best = golden_cut(phi, 0, 100,1e-5)
        # print(step_best)
        delta=step_best*grad_f
        print(delta)
        # delta=[x[i] - step_best * grad_f[i] for i in range(len(x))]
        x_last=x
        x = x-delta
        x_list.append(x)
        y_list.append(f(x))
        # print("CURRENT DOT:", x1.evalf(6))
        # print(x1,x0)
        # print(delta,np.linalg.norm(x))
        condition=np.linalg.norm(delta)/max(1,np.linalg.norm(x_last))
        # condition = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        print(k,step_best,delta)
        k += 1
        # print(k)
        # rel3=change(var,x0)
        # min_f=f.subs(rel3)
        # print(min_f[0,0].evalf(6))
    # print(x0.evalf(6))
    # rel3 = change(var, x0)
    # min_f = f.subs(rel3)
    # print(min_f[0, 0].evalf(6))
    print(f(x))
    print("Total iterations:", k)
    return x_list,y_list


# x, y, z, w = sympy.symbols('x,y,z,w')
# w=(x-5)**2+(y+4)**2+4*(z-6)**2
# var=sympy.Matrix([x,y,z])
# f=sympy.Matrix([w])
# x0=sympy.Matrix([1000,1000,1000])
# SteepestDescent(f,x0,var,1e-6)

# w = 100*(x-y**2)**2+(1-y)**2
# var = sympy.Matrix([x, y])
# f = sympy.Matrix([w])
# x0 = sympy.Matrix([0, 0])
# SteepestDescent(f, x0, var, 1e-2)

# w=x**2+(y**2)/20
# var = sympy.Matrix([x, y])
# f = sympy.Matrix([w])
# x0 = sympy.Matrix([1,1])
# SteepestDescent(f, x0, var, 1e-6)

# f=input('Plz input the function:\n')
# var=list(input('Plz input the varibles of the funcion:\n').split())
# # f=vectorize_func(f)
# # f=sympify(f)
# # f=lambdify(var,f,'numpy')
# x_0=list(map(float,input("Plz input the initial point:\n").split()))
# eps=int(input("Plz input the accuracy you want for the solution, if it's 1e-4, input 4 instead:\n"))
# eps=power(10,-eps)
# print(x_0)
# print(f(1,1))
# print(f(x_0))
x_list,y_list=SteepestDescent(f,x_0,1e-4)
print(x_list,y_list)
# w=(x**2)/5+(y**2)/10+sympy.sin(x+y)
# var = sympy.Matrix([x, y])
# f = sympy.Matrix([w])
# x0 = sympy.Matrix([0,0])
# SteepestDescent(f, x0, var, 1e-4)