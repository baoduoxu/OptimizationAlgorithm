import math

def f(x):
    # return math.pow(x,2)+4*math.cos(x)
    return (x-6)**2+math.log(x)+math.exp(x)

l=float(input())
r=float(input()) #初始化区间
# phi=(math.sqrt(5) -1)/2 #黄金分割比例
rho=(3-math.sqrt(5))/2
e=float(input()) #精度
# a=l
# b=r
k=1 #迭代次数
# print(l,r,e)
a=l+(r-l)*rho
b=r-(r-l)*rho
while(math.fabs(l-r)>e):
    print("Iterations:",k)
    print("a=%.4f" % a)
    print("b=%.4f" % b)
    print("f(a)=%.4f" % f(a))
    print("f(b)=%.4f" % f(b))
    if f(a)>f(b):
        l=a
        a=b
        b=r-(r-l)*rho
    else:
        r=b
        b=a
        a=l+(r-l)*rho
    # str1="The new section is [{},{}]"
    print("The new section is","[","%.4f" % l ,",","%.4f" % r,"]")
    k+=1