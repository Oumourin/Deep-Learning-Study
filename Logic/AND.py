import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    y = np.array([0.5, 0.5])
    b = -0.7
    tmp = sum(x*y) + b
    if tmp <= 0:
        return 0
    else:
        return 1
