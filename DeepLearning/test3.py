dict = {100: "sigmoid", 4: "ReLU"}

import numpy as np

X = np.random.uniform(-1, 1, (2, 4))
print(X)
a = np.array([0, 1, 2, 0])
en = np.arange(a.shape[0])
print(en)
print(en[en > 0])
print(np.random.choice(en[en > 0], 2, replace=False))
a[a > 1] = 3
print(a)

if 3:
    print("ok")
