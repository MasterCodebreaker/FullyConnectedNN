import numpy as np
from utils import mse, d_mse, softmax, d_softmax

# print(np.arange(0, 10, dtype=int))


# print(d_se(np.array([1, 2]), np.array([0, 3])))

# print(list(range(10)))
X = np.array(
    [[1, 0.9, -41, 0], [0.1, 0, 1, 0], [0.9, 0, 1, 0.8], [0, 1, 0, 0], [9, 1, 2, 3]]
)
Y = np.array([[0.9, 0, 1, 0.8], [0, 1, 0, 0]])
# print(X.shape)
# print(np.sum(X, axis=0) / X.shape[0])
# print(np.sum(np.sum(X, axis=1)))
# print(d_sme(X, Y))
# print(1 - True, " ok")
a = X
print(X)
print(softmax(X))
print(softmax(X).sum(axis=1))
