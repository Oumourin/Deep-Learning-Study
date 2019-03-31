import numpy as np


# X = np.array([1, 2])
# print(X.shape)
# W = np.array([[1, 3, 5], [2, 4, 6]])
# print(W)
# print(W.shape)
# Y = np.dot(X, W)
# print(Y)


# sigmoid函数
def sigmoid(x):
    return 1/(1 + np.exp(x))


# 恒等函数
def identity_function(x):
    return x


# softMax函数
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 第一层实现
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

# 第二层实现
W2 = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 第三层实现
W3 = np.array([
    [0.1, 0.3],
    [0.2, 0.4]
])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y)

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)


# print(X.shape)
# print(W1.shape)
# print(B1.shape)
