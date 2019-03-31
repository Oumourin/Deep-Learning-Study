import numpy as np
import matplotlib.pylab as plt


# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0


# 将numpy中的bool类型转换为int类型
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
y1 = step_function(x)
plt.plot(x, y, label="Sigmoid")
plt.plot(x, y1, linestyle='--', label="Step_Function")
plt.ylim(-0.1, 1.1)
plt.show()



