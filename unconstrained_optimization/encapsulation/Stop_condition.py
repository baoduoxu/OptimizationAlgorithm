import numpy as np
def stop_condition(delta, x, grad_f,condition_type):
    if condition_type==1: #梯度范数
        return np.linalg.norm(grad_f)
    elif condition_type==2: #迭代点的相对变化
        return np.linalg.norm(delta)/max(1, np.linalg.norm(x))
    else:
        raise UnicodeEncodeError