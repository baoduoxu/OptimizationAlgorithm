import numpy as np
from simplex_module import simplex

max_iter=100
eps=1e-6 #浮点数比较的精度, 对于某些特定的问题，使用该精度仍然会出现某些意想不到的问题，会将某些是整数的解误判为非整数  

def cutting_plane(A,b,c,DIS):
    _,origin_num=np.shape(A)
    _,isv,x,_=simplex(A,b,c)
    base = np.squeeze(list(np.where(isv == 1)))
    non_base = np.squeeze(list(np.where(isv == 0)))
    A2,b2,c2=A,b,c # 初始化
    idx=0 
    i=0
    while not all(np.fabs(x[i]-int(x[i]))<=eps for i in DIS):
        if i>max_iter:return 'dead loop!'
        B,D=A2[:,base],A2[:,non_base]# 获取基变量与非基变量对应的矩阵
        D=np.linalg.inv(B)@D
        idx=0
        for i in DIS: 
            if np.fabs(x[i]-int(x[i]))>eps: #对于不是整数的解
                m,n=np.shape(A2)
                A2=np.concatenate((A2,np.zeros((m,1))),axis=1)
                A2=np.concatenate((A2,np.zeros((1,n+1))),axis=0) # A2增加一行一列
                A2[m][n]=1 
                def dif(x):return np.floor(x)-x
                A2[m,non_base]=dif(D[idx,]) # 新加的那一行中，基变量的系数为零
                idx+=1
                print(f'本次添加的割平面为:{A2[m,:]},{np.round(np.floor(x[i])-x[i],decimals=6)}')
                b2=np.concatenate((b2,[np.floor(x[i])-x[i]])) #割平面等式右边未b_i的小数部分
                c2=np.concatenate((c2,[0])) #添加新变量，新变量在目标函数中的系数未0
        opt_val,isv,x,_=simplex(A2,b2,c2)
        x=np.round(x,decimals=6)
        print(f'添加割平面后的解为:{x}\n')
        base = np.squeeze(list(np.where(isv == 1)))
        non_base = np.squeeze(list(np.where(isv == 0)))
        i+=1
    return x[0:origin_num],opt_val

m,n=map(int,input().split())
A=[]
for i in range(m):
    row=list(map(float,input().split()))
    A.append(row)
b=list(map(float,input().split()))
c=list(map(float,input().split()))
A = np.array(A)
b = np.array(b)
c = np.array(c) 
DIS=[0,1]
opt_sol,opt_val=cutting_plane(A,b,c,DIS)
opt_val=np.round(opt_val,decimals=3)
print(f'最优解为{opt_sol},最优值为{opt_val}')