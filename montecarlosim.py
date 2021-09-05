import numpy as np
import matplotlib.pyplot as plt

n = 10000
stones = np.random.rand(n, 2)
plt.plot(stones[:, 0], stones[:, 1], '.r')
count = 0
for stone in stones:
    if stone[0]**2 + stone[1]**2 <= 1:
        plt.plot(stone[0], stone[1], '.g')
        count += 1
monte = count / n
print(f"Estimate Pi = {4*monte}")
print(f"Real Pi = {np.pi}")
print(f"Error = {abs(4*monte - np.pi) / np.pi * 100} %")
plt.show()
