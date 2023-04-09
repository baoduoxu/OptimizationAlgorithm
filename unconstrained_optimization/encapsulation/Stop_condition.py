import numpy as np
def stop_condition(delta, x):
    return np.linalg.norm(delta)/max(1, np.linalg.norm(x))