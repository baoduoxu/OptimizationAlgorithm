import math

def g(x):
    return math.pow(x,2)+4*math.cos(x)

l=float(input())
r=float(input()) #初始化区间
e=float(input()) #精度
epsilon=float(input()) #微扰
F=[1,1]
N=1
while True: #确定迭代次数N
    F.append(F[N]+F[N-1])
    if F[N+1]>=(2*epsilon+1)*(r-l)/e:
        break
    N+=1
print("N=",N)
rho=1-F[N]/F[N+1]
a=l+(r-l)*rho
b=r-(r-l)*rho
for i in range(N):
    print("Iterations:",i+1)
    print("rho=%.4f" % rho)
    print("a=%.4f" % a)
    print("b=%.4f" % b)
    print("f(a)=%.4f" % g(a))
    print("f(b)=%.4f" % g(b))
    if i==N-2:#下一次的rho
        rho=0.5-epsilon
    else:
        rho=1-F[N-i-1]/F[N-i]
    if g(a)>g(b):
        l=a
        a=b
        b=r-(r-l)*rho
    else :
        r=b
        b=a
        a=l+(r-l)*rho
    print("The new section is","[","%.4f" % l ,",","%.4f" % r,"]")
    
