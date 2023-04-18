import numpy as np
# np.set_printoptions(precision=3)
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)


def construct_simplex_table(A, b, c, isv, m, n):
    val = 0
    base = np.squeeze(list(np.where(isv == 1)))
    non_base = np.squeeze(list(np.where(isv == 0)))
    table = np.zeros((m+1, n+1))
    # 把A中base对应的列抽出来, 组成B, 求逆后插回去, 并将最后一列更新为B^{-1}b
    B = A[:, base]
    D = A[:, non_base]
    c = np.reshape(c, (-1, n))  # 将c转化为行向量
    c_B = np.squeeze(c[:, base])
    c_D = np.squeeze(c[:, non_base])
    inv_B = np.linalg.inv(B)
    B = np.identity(m)
    b = inv_B@b
    D = inv_B@D
    c_D = c_D-c_B@D
    val = -c_B@b
    c_B = np.reshape(np.array([0.]*m), (-1, len(c_B)))  # 行向量
    c_D = np.reshape(c_D, (-1, len(c_D)))
    col_base = np.concatenate((B, c_B), axis=0)  # 按列拼, c_B下面
    col_non_base = np.concatenate((D, c_D), axis=0)
    col_val = np.reshape(np.concatenate(
        (b, np.array([val])), axis=0), (-1, m+1))  # 得到列向量
    for i in range(len(base)):
        table[:, base[i]] = col_base[:, i]
    for i in range(len(non_base)):
        table[:, non_base[i]] = col_non_base[:, i]
    table[:, n] = col_val
    for i in range(m+1):
        for j in range(n+1):
            if np.fabs(table[i][j]) < 1e-8:
                table[i][j] = 0
    return table

def elementary_transform(table, p, q):
    M, N = np.shape(table)
    for i in range(M):
        if i != p:
            k = -table[i][q]/table[p][q]
            for j in range(N):
                table[i][j] += k*table[p][j]
    master = table[p][q]
    if master != 1:
        for j in range(N):
            table[p][j] /= master
    return table


def find_master(table, m, n):  # 返回主元的下标, 同时找到出基列
    p, q, r = -1, -1, -1
    for j in range(n):
        if table[m][j] < 0:
            q = j
            break
    tmp = 0x7f7f7f7f
    for i in range(m):
        if table[i][q] > 0:  # 应该要用浮点数
            ratio = table[i][n]/table[i][q]
            if tmp > ratio:
                tmp = ratio
                p = i
    for j in range(n):
        if np.fabs(table[p][j]-1) <= 1e-8 and np.count_nonzero(table[:, j]) == 1:
            r = j
            break
    return p, q, r


# def get_base(isv):
    base = []
    non_base = []
    for i in range(len(isv)):
        if isv[i] == 0:
            non_base.append(i)
        if isv[i] == 1:
            base.append(i)
    return base, non_base


def get_solution(table, isv):  # 从最终的单纯形表和基变量isv得到解
    M, N = np.shape(table)
    cor = list(np.where(table[:, np.squeeze(list(np.where(isv == 1)))] == 1))
    cor = list(zip(cor[0], cor[1]))
    sol = [0]*(N-1)
    for i in range(M-1):
        sol[cor[i][1]] = table[:, N-1][cor[i][0]]
    return sol


def initial_base(A, b, c, m, n):  # 构造人工问题找基本解, 返回基变量的下标
    A = np.concatenate((A, np.identity(m)), axis=1)
    c_prime = np.array([0]*n+[1]*m)
    isv_art = c_prime
    opt_val, isv_art, _, table = simplex(
        A, b, c_prime, isv_art)  # 用单纯形法求解人工问题, 得到基变量
    if opt_val > 0:
        return 'No solution'
    else:
        for i in range(n, m+n):
            if isv_art[i] == 1:  # 找人工变量中的基变量
                idx = np.where(table[:, i] == 1)  # 基列中1所在行
                for j in range(n):
                    if isv_art[j] == 0 and table[idx][j] != 0:  # 相当于第j列入基, 第i列出基
                        isv_art[j] = 1
                        isv_art[i] = 0
                        table = elementary_transform(table, idx, j)
                        print('ini:', table)
        # 返回人工问题求解后的没有去掉人工变量的单纯形表和基变量
        val = np.reshape(table[:, m+n], (-1, 1))
        table = np.concatenate((table[:, :n], val), axis=1)
        table[m:, :n] = c
        return table, isv_art[:n]


def update_checking_num(table, isv):
    M, _ = np.shape(table)
    # print(isv)
    for i in range(len(isv)):
        if isv[i] == 1:
            row = np.squeeze(np.where(table[:M-1, i] == 1.))
            table = elementary_transform(table, row, i)
    return table


def simplex(A, b, c, isv=0):  # 标准化的线性规划问题, 参数为c, A, b
    m, n = np.shape(A)
    if type(isv) != int:
        print('下面是对人工问题的求解:')
        table = construct_simplex_table(A, b, c, isv, m, n)
        print('初始单纯形表为:')
        print(table)
    else:
        table, isv = initial_base(A, b, c, m, n)  # 人工问题初始化基变量下标
        print('下面是通过人工问题找到基本解后对原问题的解,此时的单纯形表为:')
        table = update_checking_num(table, isv)
        print(table)
    k=0
    while True:
        k+=1
        p, q, r = find_master(table, m, n)  # 在非基变量中寻找主元,q为进基列,r为出基列
        if q == -1:
            print('此时检验数均非负,找到最优解.')
            return -table[m][n], isv, get_solution(table, isv), table
        if p == -1:
            print('Oops! There\'s no optimal solution for this LP.')
            return 'No solution!'
        isv[q] = 1
        isv[r] = 0
        print(f'第{k}轮迭代, 主元的坐标为({p},{q}),出基列为{r},进基列为{q},换基操作后的单纯形表为:')
        table = elementary_transform(table, p, q)  # 进行初等变换, 即相当于换基操作
        print(table)

# print()
# m,n=map(int,input().split())
# A=[]
# for i in range(m):
#     row=list(map(int,input().split()))
#     A.append(row)
# A=np.array(A)
# b=list(map(int,input().split()))
# c=list(map(int,input().split()))


# A = [[3, 3, 1, 0, 0], [4, -4, 0, 1, 0], [2, -1, 0, 0, 1]]
# b = [30, 16, 12]
# c = [-3, -1, 0, 0, 0]
# A = np.array(A)
# b = np.array(b)
# c = np.array(c)

# m,n=map(int,input().split())
# A=[]
# for i in range(m):
#     row=list(map(int,input().split()))
#     A.append(row)
# A=np.array(A)
# b=list(map(int,input().split()))
# c=list(map(int,input().split()))
# A = np.array([[1, 2, 0, 1, 0, 0],
#     [2, 1, 0, 0, 1, 0],
#     [0, 1, 1, 0, 0, 1],
#     [1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0],
# ])

# b = np.array([10, 10, 10, 6, 2])

# c = np.array([-2, -3, -1, 0, 0, 0])

A=[[2,-1,1,0,0],[2,1,0,-1,0],[1,2,0,0,1]]
b=[8,2,10]
c=[1,-3,1,0,0]
opt_val,base,opt_sol,simplex_table=simplex(A, b, c)
print(f'最优解为{opt_sol},最优值为{opt_val}')
# print(opt_val,opt_sol)
print('最终的单纯形表为:')
print(simplex_table)

# 3 5
# 3 3 1 0 0
# 4 -4 0 1 0
# 2 -1 0 0 1
# 30 16 12
# -3 -1 0 0 0

# 3 5
# 2 -1 1 0 0
# 2 1 0 -1 0
# 1 2 0 0 1
# 8 2 10
# 1 -3 1 0 0

# 3 5
# 0 5 1 0 0
# 6 2 0 1 0
# 1 1 0 0 1
# 15 24 5
# -2 -1 0 0 0

# import numpy as np
# # np.set_printoptions(precision=3)
# np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
# global flag
# flag=0

# def construct_simplex_table(A,b,c,isv,m,n):
#     val=0
#     # global table
#     # base,non_base=get_base(isv)
#     # print(isv)
#     base=np.squeeze(list(np.where(isv==1)))
#     non_base=np.squeeze(list(np.where(isv==0)))
#     table=np.zeros((m+1,n+1))
#     # table=np.array([]) #初始化单纯形表
#     # 把A中base对应的列抽出来, 组成B, 求逆后插回去, 并将最后一列更新为B^{-1}b
#     B=A[:,base]
#     D=A[:,non_base]
#     print(base,non_base,B,D)
#     c=np.reshape(c,(-1,n)) #将c转化为行向量
#     c_B=np.squeeze(c[:,base])
#     c_D=np.squeeze(c[:,non_base])
#     # for idx in base:
#     #     B=np.append(B,A[idx],axis=0)
#     #     c_B=np.append(c_B,[c[idx]],axis=0)
#     #     isv[idx]=1
#     # for i in range(n):
#     #     if isv[i]==0:
#     #         D=np.append(D,A[i],axis=0)
#     #         c_D=np.append(c_D,[c[i]],axis=0)
#     inv_B=np.linalg.inv(B)
#     print(inv_B)
#     B=np.identity(m)
#     b=inv_B@b
#     D=inv_B@D
#     c_D=c_D-c_B@D
#     val=-c_B@b
#     c_B=np.reshape(np.array([0.]*m),(-1,len(c_B))) #行向量
#     c_D=np.reshape(c_D,(-1,len(c_D)))
#     # print(B,b,D,c_D,val,c_B,sep='\n')
#     #重新拼接, 先竖着拼
#     col_base=np.concatenate((B,c_B),axis=0) #按列拼, c_B下面
#     col_non_base=np.concatenate((D,c_D),axis=0)
#     # print(b,val,np.concatenate((b,np.array([val])),axis=1))
#     col_val=np.reshape(np.concatenate((b,np.array([val])),axis=0),(-1,m+1)) #得到列向量
#     for i in range(len(base)):
#         table[:,base[i]]=col_base[:,i]
#     for i in range(len(non_base)):
#         table[:,non_base[i]]=col_non_base[:,i]
#     table[:,n]=col_val
#     for i in range(m+1):
#         for j in range(n+1):
#             if np.fabs(table[i][j])<1e-8:
#                 table[i][j]=0
#     return table
#     # # 再根据下标横着拼, 拼接与赋值谁更耗时间?
#     # for i in range(n):
#     #     idx1=0 #用于基变量
#     #     idx2=0 #用于非基变量
#     #     if isv[i]==1: #表示基变量
#     #         table=np.append(table,col_base[idx1],axis=0)
#     #         idx1+=1
#     #     if isv[i]==0:
#     #         table=np.append(table,col_non_base[idx2],axis=0)
#     #         idx2+=1
#     # table=np.append(table,col_val,axis=0)
#     # return table


# def elementary_transform(table,p,q):
#     M,N=np.shape(table)
#     for i in range(M):
#         if i!=p:
#             k=-table[i][q]/table[p][q]
#             for j in range(N):
#                 table[i][j]+=k*table[p][j]
#                 # if np.fabs(table[i][j])<1e-8:
#                 #     table[p][j]=0
#     master=table[p][q]
#     if master!=1:
#         for j in range(N): 
#             # print(table[p][j],sep=',')
#             table[p][j]/=master
#             # if np.fabs(table[p][j])<1e-8:
#             #     table[p][j]=0
#         # print('ele:',table)
#     return table

# # def change_base(q,r): #换基操作,q为进基列,r为出基列 

# def find_master(table,m,n): #返回主元的下标, 同时找到出基列
#     q=-1
#     r=-1#出基列
#     for j in range(n):
#         if table[m][j]<0:
#             q=j
#             break
#     tmp=0x7f7f7f7f
#     p=-1
#     for i in range(m):
#         if table[i][q]>0: #应该要用浮点数
#             ratio=table[i][n]/table[i][q]
#             if tmp>ratio:
#                 tmp=ratio
#                 p=i
#     for j in range(n):
#         # print(np.count_nonzero(table[:,j]))
#         # print(np.fabs(table[p][j]-1)<1e-8,np.count_nonzero(table[:,j]),m,np.count_nonzero(table[:,j])==m)
#         if np.fabs(table[p][j]-1)<=1e-8 and np.count_nonzero(table[:,j])==1:
#             r=j
#             break
#     # 在第p行找到第j列, 满足table[p][j]==1且第j列只有1个1
#     # print(r)
#     return p,q,r
# # 问题形式为
# # min c^Tx
# # s.t. Ax=b
# def get_base(isv):
#     base=[]
#     non_base=[]
#     for i in range(len(isv)):
#         if isv[i]==0:
#             non_base.append(i)
#         if isv[i]==1:
#             base.append(i)
#     return base,non_base

# def get_solution(table,isv): #从最终的单纯形表和基变量isv得到解
#     M,N=np.shape(table)
#     # print(isv)
#     # base=np.squeeze(list(np.where(isv==1)))
#     # print(base,table[:,base])
#     cor=list(np.where(table[:,np.squeeze(list(np.where(isv==1)))]==1))
#     cor= list(zip(cor[0], cor[1]))
#     sol=[0]*(M-1)
#     # print(cor)
#     for i in range(M-1):
#         # print(table[:,N-1])
#         sol[cor[i][1]]=table[:,N-1][cor[i][0]]
#     return sol

# def initial_base(A,b,c,m,n): #构造人工问题找基本解, 返回基变量的下标
#     # global flag
#     # flag=1
#     # print(A,np.identity(m))
#     A=np.concatenate((A,np.identity(m)),axis=1)
#     c_prime=np.array([0]*n+[1]*m)
#     isv_art=c_prime
#     opt_val,isv_art,_,table=simplex(A,b,c_prime,isv_art) # 用单纯形法求解人工问题, 得到基变量
#     # print('ini',table)
#     if opt_val>0:
#         return 'No solution'
#     else:
#         for i in range(n,m+n):
#             if isv_art[i]==1: # 找人工变量中的基变量
#                 idx=np.where(table[:,i]==1) #基列中1所在行
#                 for j in range(n):
#                     if isv_art[j]==0 and table[idx][j]!=0: # 相当于第j列入基, 第i列出基
#                         isv_art[j]=1
#                         isv_art[i]=0
#                         table=elementary_transform(table,idx,j)
#                         print('ini:',table)
#         # 返回人工问题求解后的没有去掉人工变量的单纯形表和基变量
#         val=np.reshape(table[:,m+n],(-1,1))
#         # print(table[:,:n],val)
#         # print(isv_art[:n])
#         table=np.concatenate((table[:,:n],val),axis=1)
#         # print(table,np.shape(table),c,table[m:,:n])
#         table[m:,:n]=c
#         # print('after',table)
#         return table,isv_art[:n]
        
#         # return get_base(var)

# def update_checking_num(table,isv):
#     M,_=np.shape(table)
#     print(isv)
#     for i in range(len(isv)):
#         if isv[i]==1:
#             # print(table[:M-1,i])
#             # print()
#             row=np.squeeze(np.where(table[:M-1,i]==1.))
#             table=elementary_transform(table,row,i)
#     return table

# def simplex(A,b,c,isv=0): # 标准化的线性规划问题, 参数为c, A, b
#     m,n=np.shape(A)
#     if type(isv)!=int:
#         # base,non_base=get_base(isv)
#         table=construct_simplex_table(A,b,c,isv,m,n)
#     # if flag==0: #flag=0表示还没有求解人工问题
#     else:
#         print(c)
#         table,isv=initial_base(A,b,c,m,n) #人工问题初始化基变量下标
#         table=update_checking_num(table,isv)
#         # flag=1
#     # global isv
#         # base,non_base=get_base(isv) 
#     # isv=[0]*n #用于监测基变量的变化,1为基变量,0为非基变量
#     # non_base=[] #非基变量下标, base和non_base用于取B和D和拼接
#     # for x in base:
#     #     isv[x]=1
#     # for i in range(n):
#     #     if isv[i]==0:
#     #         non_base.append(i)
#          #得到初始单纯形表
#     print(table)
#     while True:
#         # print('base: ',isv)
#         p,q,r=find_master(table,m,n) #在非基变量中寻找主元,q为进基列,r为出基列
#         print(p,q,r)
#         if q==-1:
#             print('Find the optimal solution!')
#             return -table[m][n],isv,get_solution(table,isv),table
#         if p==-1:
#             print('Oops! There\'s no optimal solution for this LP.')
#             return 'No solution!'
#         isv[q]=1
#         isv[r]=0
#         table=elementary_transform(table,p,q)#进行初等变换, 即相当于换基操作
#         print(table)

# # print()
# # m,n=map(int,input().split())
# # A=[]
# # for i in range(m):
# #     row=list(map(int,input().split()))
# #     A.append(row)
# # A=np.array(A)
# # b=list(map(int,input().split()))
# # c=list(map(int,input().split()))

# A=[[3,3,1,0,0],[4,-4,0,1,0],[2,-1,0,0,1]]
# b=[30,16,12]
# c=[-3,-1,0,0,0]
# A=np.array(A)
# b=np.array(b)
# c=np.array(c)

# # m,n=map(int,input().split())
# # A=[]
# # for i in range(m):
# #     row=list(map(int,input().split()))
# #     A.append(row)
# # A=np.array(A)
# # b=list(map(int,input().split()))
# # c=list(map(int,input().split()))
# # A = np.array([[1, 2, 0, 1, 0, 0],
# #     [2, 1, 0, 0, 1, 0],
# #     [0, 1, 1, 0, 0, 1],
# #     [1, 1, 0, 0, 0, 0],
# #     [0, 0, 1, 0, 0, 0],
# # ])

# # b = np.array([10, 10, 10, 6, 2])

# # c = np.array([-2, -3, -1, 0, 0, 0])

# # A=[[2,-1,1,0,0],[2,1,0,-1,0],[1,2,0,0,1]]
# # b=[8,2,10]
# # c=[1,-3,1,0,0]
# print(simplex(A,b,c))

# # 3 5
# # 3 3 1 0 0
# # 4 -4 0 1 0
# # 2 -1 0 0 1
# # 30 16 12
# # -3 -1 0 0 0

# # 3 5
# # 2 -1 1 0 0
# # 2 1 0 -1 0
# # 1 2 0 0 1
# # 8 2 10
# # 1 -3 1 0 0

# # 3 5
# # 0 5 1 0 0
# # 6 2 0 1 0
# # 1 1 0 0 1
# # 15 24 5
# # -2 -1 0 0 0