import math
import numpy as np
from numpy import *
import sympy
from sympy import sympify

rho = (3-math.sqrt(5))/2

def change(var, val):
    l = []
    it1 = iter(var)
    it2 = iter(val)
    k = var.shape[0]
    for i in range(k):
        l.append(tuple((next(it1), next(it2))))
    return l


def golden_cut(f, l, r, var, eps):
    a = l+(r-l)*rho
    b = r-(r-l)*rho
    while (math.fabs(l-r) > eps):
        f_a = f.subs([(var, a)])
        f_b = f.subs([(var, b)])
        if f_a[0] > f_b[0]:
            l = a
            a = b
            b = r-(r-l)*rho
        else:
            r = b
            b = a
            a = l+(r-l)*rho
    return a


def SteepestDescent(f, x0, var, eps):
    grad_f = f.jacobian(var)  # 雅可比矩阵
    k = 0
    con = 1
    while con > eps:
        rel1 = change(var, x0)
        d = grad_f.subs(rel1).T
        alpha = sympy.symbols('alpha')
        rel2 = change(var, x0-alpha*d)
        phi = f.subs(rel2)
        alpha0 = golden_cut(phi, 0, 100, alpha, 1e-5)
        x1 = x0-alpha0*d
        print("CURRENT DOT:", x1.evalf(6))
        tmp = x1
        x1 = x0
        x0 = tmp
        con = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        print(con.evalf(6))
        # print(sympy.Matrix.norm(x1-x0))
        print(k,alpha0,x1-x0)
        k += 1
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

w=x**2+y**2
var = sympy.Matrix([x, y])
f = sympy.Matrix([w])
x0 = sympy.Matrix([1,1])
SteepestDescent(f, x0, var, 1e-6)

# func=input('Plz input the function you want to test with steepest descent method:')
# f=sympify(func)
# # print(f)
# # w=(x**2)/5+(y**2)/10+sympy.sin(x+y)
# var = sympy.Matrix([x, y])
# # f = sympy.Matrix([w])
# x0 = sympy.Matrix([0,0])
# SteepestDescent(f, x0, var, 1e-4)