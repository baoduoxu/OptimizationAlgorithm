import numpy as np
from simplex import simplex

print('请确保输入标准形式的线性规划, 依次输入系数矩阵的行和列, 系数矩阵, 等式右端向量,以及目标函数的系数向量')
m,n=map(int,input().split())
A=[]
for i in range(m):
    row=list(map(int,input().split()))
    A.append(row)
A=np.array(A)
b=list(map(int,input().split()))
c=list(map(int,input().split()))
opt_val,base,opt_sol,simplex_table=simplex(A, b, c)
print(f'最优解为{np.round(opt_sol,3)},最优值为{np.round(opt_val,3)}')
print('最终的单纯形表为:')
print(simplex_table)


# 下面是一些测试数据
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

# A = np.array([[1, 2, 0, 1, 0, 0],
#     [2, 1, 0, 0, 1, 0],
#     [0, 1, 1, 0, 0, 1],
#     [1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0],
# ])
# b = np.array([10, 10, 10, 6, 2])
# c = np.array([-2, -3, -1, 0, 0, 0])

# A = [[3, 3, 1, 0, 0], [4, -4, 0, 1, 0], [2, -1, 0, 0, 1]]
# b = [30, 16, 12]
# c = [-3, -1, 0, 0, 0]
# A = np.array(A)
# b = np.array(b)
# c = np.array(c)

# A=[[2,-1,1,0,0],[2,1,0,-1,0],[1,2,0,0,1]]
# b=[8,2,10]
# c=[1,-3,1,0,0]