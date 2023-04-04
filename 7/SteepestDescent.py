import math
import numpy as np
from numpy import *
import sympy


def change(var, val):
    l = []
    it1 = iter(var)
    it2 = iter(val)
    k = var.shape[0]
    for i in range(k):
        l.append(tuple((next(it1), next(it2))))
    return l


def golden_cut(f, l, r, var, eps):
    # print(f)
    rho = (3-math.sqrt(5))/2
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
        f_a = f.subs([(var, a)])
        # print(f_a)
        f_b = f.subs([(var, b)])
        # print(f_b)
        if f_a[0, 0] > f_b[0, 0]:
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


def SteepestDescent(f, x0, var, eps):
    grad_f = f.jacobian(var)  # 雅可比矩阵
    # d=grad_f.subs(var,x0) #初始值
    # s=var.shape
    # x1=sympy.Matrix.zeros(s[0],1)
    # print(x1.shape)
    # print(x0.shape)
    # con=sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0),1)
    # d=grad_f.subs(var,x0)
    # print(type(d))
    k = 0
    con = 1
    while con > eps:
        rel1 = change(var, x0)
        d = grad_f.subs(rel1).T
        # print(d.shape)
        alpha = sympy.symbols('alpha')
        rel2 = change(var, x0-alpha*d)
        phi = f.subs(rel2)
        alpha0 = golden_cut(phi, 0, 100, alpha, 1e-5)
        # print("alpha0=",alpha0)
        x1 = x0-alpha0*d
        print("CURRENT DOT:", x1.evalf(6))
        tmp = x1
        x1 = x0
        x0 = tmp
        # print(x1,x0)
        con = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        print(con.evalf(6))
        k += 1
        # print(k)
        # rel3=change(var,x0)
        # min_f=f.subs(rel3)
        # print(min_f[0,0].evalf(6))
    print(x0.evalf(6))
    rel3 = change(var, x0)
    min_f = f.subs(rel3)
    print(min_f[0, 0].evalf(6))
    print("Total iterations:", k)


x, y, z, w = sympy.symbols('x,y,z,w')
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

w=(x**2)/5+(y**2)/10+sympy.sin(x+y)
var = sympy.Matrix([x, y])
f = sympy.Matrix([w])
x0 = sympy.Matrix([0,0])
SteepestDescent(f, x0, var, 1e-4)