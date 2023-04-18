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
            return -1
        isv[q] = 1
        isv[r] = 0
        print(f'第{k}轮迭代, 主元的坐标为({p},{q}),出基列为{r},进基列为{q},换基操作后的单纯形表为:')
        table = elementary_transform(table, p, q)  # 进行初等变换, 即相当于换基操作
        print(table)