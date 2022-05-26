# Libraries
import numpy as np
import matplotlib. pylab as plt

# Number of data
N = 10000

# Generate data
# Choose the line x=t, y=t, z=t, if y›x, then y=1, else y=0
X = np.random.randint(0, 250, (N, 3))
Y = np.ones(N)

# Label the data
Y[X[:, 0] <= X[:, 1]] = 0

# Flip the labels make the data non-separable
for i in Y:
    r = np.random.uniform(0, 1)
    if r <= 0.1:
        if i == 0:
            i = 1
        else:
            i = 0

print(len(X))
print(Y)
