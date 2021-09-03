# Feedforward Backpropagation Neural Network
# 05/06/2019
# Copyright (C) 2019 Parinya Sanguansat
import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x) # x is sigmoid


class FFBP:
    def __init__(self, X, T, hidden, lr=1e-1, alpha=1e-2, eps=1e-3, epochs=float('inf'), f=sigmoid, df=d_sigmoid):
        self.X = X
        self.T = T
        self.hidden = hidden
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.epochs = epochs
        self.f = f
        self.df = df

        self.d_in = self.X.shape[1]
        if len(self.T.shape) == 1:
            self.T = np.reshape(self.T, (len(X), 1))
        del X, T
        self.d_out = self.T.shape[1]

        self.error = []
        self.W = []
        self.init_weights()
        self.input = [None] * (len(self.W)+1)

        self.forward(self.X)
        self.backward()

    @staticmethod
    def pad_ones(input):
        return np.hstack((input, np.ones((len(input), 1))))

    def init_weights(self):
        dim = [self.d_in] + self.hidden + [self.d_out]
        for i in range(len(self.hidden)+1):
            self.W.append(np.random.rand(dim[i]+1, dim[i+1])) # add bias

    def forward(self, input):
        input = self.pad_ones(input)
        self.input[0] = input.copy()
        for i in range(len(self.W)):
            output = self.f(input @ self.W[i])
            input = self.pad_ones(output)
            self.input[i+1] = input.copy()
        return output

    def backward(self):
        epoch = 0
        dw = [0] * len(self.W)
        while True:
            epoch += 1
            output = self.forward(self.X)

            error = self.T - output
            mse = np.mean(error ** 2)
            print('epoch {}:\terror={}'.format(epoch, mse))
            self.error.append(mse)
            if mse < self.eps or epoch >= self.epochs:
                break

            # output layer
            delta = self.df(output) * error
            dw[-1] = self.lr * self.input[-2].T @ delta + self.alpha * dw[-1]
            self.W[-1] += dw[-1]
            # hidden layers
            for i in range(len(self.W)-2, -1, -1):
                delta = self.df(self.input[i+1]) * (delta @ self.W[i+1].T)
                delta = delta[:, :-1]
                dw[i] = self.lr * self.input[i].T @ delta + self.alpha * dw[i]
                self.W[i] += dw[i]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T = np.array([0, 1, 1, 0])
    hidden = [2]

    net = FFBP(X, T, hidden, lr=0.1, alpha=0.8, eps=0.001, epochs=5000, f=sigmoid, df=d_sigmoid)


    plt.plot(net.error)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show()

    #  Plot hyperplane
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    x = np.arange(0, 1.01, .01)
    y = np.arange(0, 1.01, .01)
    output = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            output[i, j] = net.forward(np.array([[x[i], y[j]]]))[0][0]
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, output, cmap=cm.cool,
                           linewidth=0, antialiased=False)
    ax.scatter(X[:, 0][T == 0], X[:, 1][T == 0], T[T == 0], color='g', s=50)
    ax.scatter(X[:, 0][T == 1], X[:, 1][T == 1], T[T == 1], color='r', s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Output')
    plt.show()
