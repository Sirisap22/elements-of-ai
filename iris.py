import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def knn(x, x_train, y_train):
    return y_train[np.sum((x_train - x) ** 2, axis=1).argmin()]

if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(url, header=None)
    i_train = np.r_[:25, 50:75, 100:125]
    i_test = np.r_[25:50, 75:100, 125:150]

    x_train = df.iloc[i_train, :-1].values
    x_test = df.iloc[i_test, :-1].values
    y_train = df.iloc[i_train, -1].values
    y_test = df.iloc[i_test, -1].values

    z_test = []
    for x in x_test:
      z_test.append(knn(x))

    '''k-NN'''
    z_test = np.array(z_test)
    accuracy = np.sum(z_test == y_test) / len(y_test) * 100
    print(accuracy)


# # fig = plt.figure().gca(projection='2d')
# # fig.scatter(df[0][:50], df[1][:50], df[2][:50], c='r')
# # fig.scatter(df[0][50:100], df[1][50:100], df[2][50:100], c='g')
# # fig.scatter(df[0][100:], df[1][100:], df[2][100:], c='b')
# # plt.show()

# plt.plot(df[0][:50], df[1][:50], '.r')
# plt.plot(df[0][50:100], df[1][50:100], '.g')
# plt.plot(df[0][100:], df[1][100:], '.b')
# plt.show()