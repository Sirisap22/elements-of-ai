import numpy as np
import matplotlib.pyplot as plt

A_shape = 2, 3, 3
A = np.zeros(A_shape, dtype=np.uint8)
# A[[0, -1], :, 0] = 255
# A[[1, -2], :, :] = 255
# A[[2, 3], :, 2] = 255
# France
A[:, 0, 1] = 85
A[:, 0, 2] = 164
A[:, 1, :] = 255
A[:, 2, 0] = 239
A[:, 2, 1] = 65
A[:, 2, 2] = 53

# Germany
B_shape = (3, 5, 3)
B = np.zeros(B_shape, dtype=np.uint8)
B[1, :, 0] = 255
B[2, :, 0] = 255
B[2, :, 1] = 204

# Japan
# (x-h)**2 + (y-k)**2 = r**2
multipler = 500
C_shape = 2*multipler, 3*multipler, 3
C = np.zeros(C_shape, dtype=np.uint8)
C[:, :, :] = 255
r = 3/5 * multipler
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        if (i-C.shape[0]/2)**2 + (j-C.shape[1]/2)**2 <= r**2:
          C[i, j, 0] = 255
          C[i, j, 1:] = 0
plt.imshow(C)
plt.show()