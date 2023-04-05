# from Optimization.8.newton_high_dim import golden_cut
# from Optimization.8.newton_high_dim import change
# from pkgutil import ImpImporter
import sympy
import newton_high_dim
import math
def HS(d,g1,g0):#Hestenes-Stiefel公式
    a=g1.T*(g1-g0)
    b=d.T*(g1-g0)
    return a[0]/b[0]

def PR(g1,g0):#Polak-Ribiere公式
    # print(g1,g0)
    a=g1.T*(g1-g0)
    # print(a)
    # print(type(a[0]))
    # print()
    b=g0.T*g0
    # print(b)
    return a[0]/b[0]

def FR(g1,g0):#Fletcher-Reeves公式
    a=g1.T*g1
    b=g0.T*g0
    return a[0]/b[0]

def conjugate_grad(f,x0,var,eps):
    gradf=f.jacobian(var).T
    print(gradf)
    # rel1=newton_high_dim.change(var,x0)
    # g0=gradf.subs(rel1)#初始方向
    # d=-g0
    con=1#停机条件初始化为1
    k=0#迭代次数
    n=var.shape#变量维数
    # print(n[0])
    while con>eps and k<20:
        print(k,n[0],k%n[0])
        if k%n[0]==0:#每n步初始化方向为当前点的梯度
            rel3=newton_high_dim.change(var,x0)
            g0=gradf.subs(rel3)
            d=-g0
        else:
            rel2=newton_high_dim.change(var,x0)
            g1=gradf.subs(rel2)#当前点的梯度
            # print("Previous gradient: g0=",g0)
            print("Current gradient: g1=",g1)
            # beta=PR(g1,g0)#g0是上一步的梯度,g1是当前的梯度
            # beta=HS(d,g1,g0)
            beta=FR(g1,g0)
            # print("beta=%.4f" % beta)
            d=-g1+beta*d
            g0=g1
        #找最优步长
        alpha = sympy.symbols('alpha')#步长变量
        rel4 = newton_high_dim.change(var, x0+alpha*d)
        # print(var.shape)
        # print(x0.shape)
        # print(d.shape)
        phi = f.subs(rel4)
        alpha0 = newton_high_dim.golden_cut(phi, 0, 100, alpha, 1e-4)#黄金分割求出最优步长
        # print("alpha0=",alpha0)
        # print("direction is: d=",d)
        x1=x0+alpha0*d#求出下一个点
        # con = sympy.Matrix.norm(x1-x0)/max(sympy.Matrix.norm(x0), 1)
        rel2=newton_high_dim.change(var,x0)
        g1=gradf.subs(rel2)#当前点的梯度
        # print("GRADIENT=",g1)
        con=sympy.Matrix.norm(g1)
        print("con = %.4f" % con)
        print("CURRENT DOT:", x1)
        # tmp=x0
        x0=x1
        # x1=tmp
        k+=1
        rel3 = newton_high_dim.change(var, x0)
        min_f = f.subs(rel3)
    print(min_f[0, 0].evalf(6))
        # print(k)
    print(x0.evalf(6))
    rel3 = newton_high_dim.change(var, x0)
    min_f = f.subs(rel3)
    print(min_f[0, 0].evalf(6))
    print("Total iterations:", k)

x, y, z, w = sympy.symbols('x,y,z,w')
# w=(x-5)**2+(y+4)**2+4*(z-6)**3
w = 100*(x-y**2)**2+(1-y)**2
# w=(3/2)*x**2+2*y**2+(3/2)*z**2+x*z+2*y*z-3*x-z
# w=x**2+y**4
# var = sympy.Matrix([x, y,z])
var=sympy.Matrix([x,y])
f = sympy.Matrix([w])
x0 = sympy.Matrix([0,0])
conjugate_grad(f, x0, var, 1e-5)