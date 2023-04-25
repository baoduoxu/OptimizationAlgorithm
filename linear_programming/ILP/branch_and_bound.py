import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import heapq
from scipy.optimize import linprog

max_iter=100
eps=1e-6 #浮点数比较的精度

def change_type(x,A2,b2,c2,flag): # 为了插入堆中不报错而要做的类型转换
    if flag==0:
        return np.array(x),np.array(A2),np.array(b2),np.array(c2)
    if flag==1:
        return x.tolist(),A2.tolist(),b2.tolist(),c2.tolist()

def branch_and_bound(A,b,c,DIS): # DIS为约束为整数的那些变量的下标
    res=linprog(c,A_eq=A,b_eq=b)
    x,y=res.x,res.fun
    var_len=len(x)
    x_best,y_best,Q=x,np.inf,[] # Q中存入线性规划问题和对应的最优解和最优值，根据最优值排序
    heapq.heappush(Q,(y,x,A,b,c))
    k=0
    while len(Q)!=0:
        if k>max_iter:
            print('死循环')
            break
        k+=1
        y,x,A2,b2,c2=heapq.heappop(Q)
        x,A2,b2,c2=change_type(x,A2,b2,c2,0)
        print(y_best,x_best,y,x)
        if all(np.fabs(x[i]-int(x[i]))<eps for i in DIS): #如果当前解均为整数解
            if y<y_best:
                x_best,y_best=x[0:n],y
        else:
            i=np.argmax([np.fabs(x[i]-round(x[i]))for i in DIS]) # 距离整数最远的那个解的下标
            m, n = np.shape(A2)
            A2_left = np.concatenate((A2, np.zeros((m, 1))), axis=1)
            A2_left = np.concatenate((A2_left, np.zeros((1, n + 1))), axis=0)
            A2_left[m][i],A2_left[m][n]=1,1
            b2_left,c2_left = np.concatenate((b2, [np.floor(x[i])])),np.concatenate((c2, [0])) # 重新构造矩阵和向量
            res=linprog(c2_left,A_eq=A2_left,b_eq=b2_left)
            x2,y2,status=res.x,res.fun,res.status
            if status==0 and y2<=y_best: #status为0表示有解，只有下界小于当前的最优解才能入队
                x2,A2_left,b2_left,c2_left=change_type(x2,A2_left,b2_left,c2_left,1)
                heapq.heappush(Q,(y2,x2,A2_left,b2_left,c2_left))
            A2_right = np.concatenate((A2, np.zeros((m, 1))), axis=1)
            A2_right = np.concatenate((A2_right, np.zeros((1, n + 1))), axis=0)
            A2_right[m][i], A2_right[m][n] = 1, -1
            b2_right, c2_right = np.concatenate((b2, [np.ceil(x[i])])), np.concatenate((c2, [0]))
            res=linprog(c2_right,A_eq=A2_right,b_eq=b2_right)
            x2,y2,status=res.x,res.fun,res.status
            if status==0 and y2<=y_best: 
                x2,A2_right,b2_right,c2_right=change_type(x2,A2_right,b2_right,c2_right,1)
                heapq.heappush(Q,(y2,x2,A2_right,b2_right,c2_right))
    return x_best[0:var_len],y_best

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
print(branch_and_bound(A,b,c,DIS))