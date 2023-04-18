本仓库对常见的最优化算法用 Python 进行了实现, 同时会对机器学习的一些算法转化成的优化模型采用一些实际问题的数据进行试验.

如要测试, 请在 `unconstrained_optimization/encapsulation/test_function.py` 文件中定义函数以及初值点. 推荐采用典型的二次型函数, Rosenbrock 函数 $f(x,y)=100(x-y^2)^2+(1-y)^2 ,$ 以及函数 $f(x,y)=\frac{x^2}{5}+\frac{y^2}{10}+\sin(x+y)$ 进行测试. 可视化的例子在 `unconstrained_optimization/first_order_method/example` 中.

目前已经实现的算法已在下文加粗.

> 目前存在的问题:
>
> 在编写时求解最优步长  $\alpha_k=\arg\min_{\alpha>0}f(x_k+\alpha d_k)$ 的算法时采用了黄金分割法, 这是不对的, 黄金分割法只适用于单峰函数, 若使用黄金分割法且初值距极小点较远时, 对于一些病态的函数需要将精度调到 $10^{-10}$ 以下且迭代几百次甚至上千次才会得到比较好的值(迭代过程中步长的变化不是单调的), 后续将采用非精确线搜进行更正. 

## 优化算法

1: 一维搜索:

- **黄金分割法**
- **斐波那契法**
- 牛顿法
- 割线法
- 非精确线搜

2: 无约束优化:

- **最速下降法**
- **共轭梯度法**
- **牛顿法及其修正**
- **拟牛顿法** (目前对于非二次型函数不收敛)

3: 有约束优化:

- 投影法
- 罚函数法

4: 线性规划:

- 两阶段单纯形法

5: 整数线性规划:

- Gomory 割平面法
- 分支定界

## 机器学习中的优化算法

给定样本 $\{(x_i,y_i)\}_{i=1}^N,x_i\in\mathbb{R}^n,$ 有下面的约束问题:

1: 多元线性回归

$$
\begin{aligned}
&\min_{w\in\mathbb{R}^n} f(w)=\frac{1}{2}\sum_{i=1}^N(w^Tx_i-y_i)^2\\
\end{aligned}
$$

2: 多元线性回归的正则化:

3: 支持向量机

$$
\begin{aligned}
&\min \|w\|^2\\
\text{s.t.}&y^{(i)}(w^Tx^{(i)}+b)\ge 1,i=1,\cdots,m\\
&w\in\mathbb{R}^n\\
&b\in\mathbb{R}
\end{aligned}
$$

$$
\begin{aligned}
&\max W(\alpha)=\sum_{i}\alpha_i-\frac{1}{2}\sum_i\sum_j\alpha_i\alpha_jy^{(i)}y^{(j)}{x^{(i)}}^Tx^{(j)}\\
\text{s.t. }&\alpha_i\ge 0,i=1,\cdots,m\\
&\sum_{i=1}^m\alpha_iy^{(i)}=0
\end{aligned}
$$

> 待更新.

