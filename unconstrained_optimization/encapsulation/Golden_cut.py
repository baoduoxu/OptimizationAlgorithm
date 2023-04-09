import math
rho = (3-math.sqrt(5))/2
def golden_cut(f, l=0, r=100, eps=1e-3):
    a = l+(r-l)*rho
    b = r-(r-l)*rho
    while (math.fabs(l-r) > eps):
        if f(a) > f(b):
            l = a
            a = b
            b = r-(r-l)*rho
        else:
            r = b
            b = a
            a = l+(r-l)*rho
    return a