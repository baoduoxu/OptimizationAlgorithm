import newton_high_dim
import sympy

def newton(f, x0, var, eps):
    con = 1
    k = 0
    HF = sympy.hessian(f, var)  # 黑塞矩阵
    grad_f = f.jacobian(var).T
    while con > eps and k<1:
        rel1 = newton_high_dim.change(var, x0)
        g = grad_f.subs(rel1)
        F = HF.subs(rel1)
        print(F)
        E=sympy.Matrix.eigenvals(F)
        _min=0
        for x in E:
            if _min<x:
                _min=x
        if _min<0:
            mu=-_min+1
        # if sympy.Matrix.det(F)!=0:
            
        #     x1 = x0-sympy.Matrix.inv(F)*g
        # else:
        #     #执行最速下降法
        # tmp = x1
        # x1 = x0
        # x0 = tmp
        # con = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        k += 1
    print(x0.evalf(6))
    rel3 = newton_high_dim.change(var, x0)
    min_f = f.subs(rel3)
    print(min_f[0, 0].evalf(6))
    print("Total iterations:", k)

x, y, z, w = sympy.symbols('x,y,z,w')
# w=(x-5)**2+(y+4)**2+4*(z-6)**2
w = 100*(x-y**2)**2+(1-y)**2
var = sympy.Matrix([x, y])
f = sympy.Matrix([w])
x0 = sympy.Matrix([0, 0])
newton(f, x0, var, 1e-6)