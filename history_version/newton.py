# from numpy import swapaxes
import sympy
import math


def change(var, val):
    l = []
    it1 = iter(var)
    it2 = iter(val)
    k = var.shape[0]
    for i in range(k):
        l.append(tuple((next(it1), next(it2))))
    return l


def golden_cut(f, l, r, var, eps):
    rho = (3-math.sqrt(5))/2
    a = l+(r-l)*rho
    b = r-(r-l)*rho
    while (math.fabs(l-r) > eps):
        f_a = f.subs([(var, a)])
        f_b = f.subs([(var, b)])
        if f_a[0, 0] > f_b[0, 0]:
            l = a
            a = b
            b = r-(r-l)*rho
        else:
            r = b
            b = a
            a = l+(r-l)*rho
    return a


def newton(f, x0, var, eps):
    con = 1
    k = 0
    HF = sympy.hessian(f, var)  # 黑塞矩阵
    grad_f = f.jacobian(var).T
    while con > eps:
        rel1 = change(var, x0)
        g = grad_f.subs(rel1)
        F = HF.subs(rel1)
        x1 = x0-sympy.Matrix.inv(F)*g
        tmp = x1
        x1 = x0
        x0 = tmp
        con = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        k += 1
    print("Loccal minimal point:",x0.evalf(6))
    rel3 = change(var, x0)
    min_f = f.subs(rel3)
    print("Local minimal value:",min_f[0, 0].evalf(6))
    print("Total iterations:", k)


x, y, z, w = sympy.symbols('x,y,z,w')
# # # w=(x-5)**2+(y+4)**2+4*(z-6)**2
# w = 100*(x-y**2)**2+(1-y)**2
# var = sympy.Matrix([x, y])
# f = sympy.Matrix([w])
# x0 = sympy.Matrix([0, 0])
# newton(f, x0, var, 1e-6)

w=(x**2)/5+(y**2)/10+sympy.sin(x+y)
var = sympy.Matrix([x, y])
f = sympy.Matrix([w])
x0 = sympy.Matrix([0.1,0.1])
newton(f, x0, var, 1e-6)