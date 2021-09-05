from os import error
import numpy as np
import matplotlib.pyplot as plt

solution = [1, 3, -4, 2, -1, 1]
n = 50
x = np.random.rand(n)
y = np.polyval(solution, x)

# plt.plot(x, y, '.')
# plt.show()

def fitness(population):
    errors = []
    for sample in population:
        k = np.polyval(sample, x)
        e = np.mean((k - y)**2)
        errors.append(e)
    return np.array(errors)


# Initialize population
n_pop = 100
degree = 5
population = np.random.rand(n_pop, degree+1)
X = np.arange(0, 1, 0.001)
while True:
    f = fitness(population)
    idx = f.argsort()
    print(f"fitness = {f[idx[0]]}")
    plt.clf()
    plt.plot(x, y, '.r')
    Y = np.polyval(population[idx[0]], X)
    plt.plot(X, Y, 'r')
    plt.draw()
    plt.pause(0.0001)
    population = population[idx]
    for i in range(50, 100):
        parents = np.random.permutation(50)[:2]
        cross_over_point = np.random.randint(0, degree+1)
        population[i] = np.concatenate((population[parents[0]][:cross_over_point], population[parents[1]][cross_over_point:]))
        if np.random.rand() > 0.9:
            mutation_point = np.random.randint(0, degree+1)
            population[i][mutation_point] += np.random.rand() * (-1)**np.random.randint(2) 