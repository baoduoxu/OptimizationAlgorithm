import autograd.numpy
import numpy as np
def f(x):  # 在这里编写函数
    return 100*(x[0]-x[1]**2)**2+(1-x[1])**2  # 初始点为[0,0]
    # return 4*(x[0]-2)**2+3*(x[1]-1)**2
    # return (x[0]**2)/5+(x[1]**2)/10+autograd.numpy.sin(x[0]+x[1])
x_0 = np.array([-2., 2.])  # 在这里定义初始点,请在写浮点数