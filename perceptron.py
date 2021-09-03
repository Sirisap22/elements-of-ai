import numpy as np

# vector base perceptron AND problem
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
Y = np.array([1, -1, -1, -1])

# pad 1s
X = np.column_stack((X, np.ones((len(X), 1), dtype=np.int8)))

X_ = X.copy()
for i in range(len(X_)):
  X_[i] = Y[i] * X[i]

# initialize W
W = X_[np.random.choice(len(X))]

# update W
counter = 0
while counter != len(X):
    counter = 0
    for x in X_:
        if np.dot(x, W) < 0:
            W = W + x
            print(W)
        else:
            counter += 1
print(W)

def f(x):
    return 1 if x >= 0 else -1
for x in X:
    print(f(np.dot(x, W)))
