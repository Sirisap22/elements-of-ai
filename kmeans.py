import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iris import knn

def kmean(k):
    # initial centroids
    idx_centroids = np.random.permutation(len(x_train))[:k]
    centroids = x_train[idx_centroids]

    while True:
        # distances
        distances = []
        for centroid in centroids:
            distances.append(np.sum((x_train - centroid) ** 2, axis=1))
        distances = np.array(distances)
        idx_groups = distances.argmin(axis=0)

        # update centroids
        centroids_old = centroids.copy()
        for i in range(len(centroids)):
            centroids[i] = np.mean(x_train[idx_groups==i], axis=0)
        plt.clf() # clear frame
        plt.plot(x_train[:, 0], x_train[:, 1], '.g')
        plt.plot(centroids[:, 0], centroids[:, 1], 'or')
        plt.draw()
        plt.pause(1)

        if np.sum(np.abs(centroids_old - centroids)) == 0:
            return centroids


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
i_train = np.r_[:25, 50:75, 100:125]
i_test = np.r_[25:50, 75:100, 125:150]

x_train = df.iloc[i_train, :-1].values
x_test = df.iloc[i_test, :-1].values
y_train = df.iloc[i_train, -1].values
y_test = df.iloc[i_test, -1].values

C = kmean(3)
L = []
for c in C:
    L.append(knn(c, x_train, y_train))
L = np.array(L)
print(C)
print(L)

z_test = []
for x in x_test:
  z_test.append(knn(x, C, L))
accuracy = np.sum(np.array(z_test) == y_test) / len(x_test) * 100
print(accuracy)