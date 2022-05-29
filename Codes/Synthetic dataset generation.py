# Libraries
import numpy as np
import matplotlib. pylab as plt
import pandas as pd

# Number of data
N = 5000

# Generate data
# Choose the line y=x, if y›x, then y=1, else y=0
X = np.random.randint(0, 250, (N, 2))
Y = np.ones(N)

# Label the data
Y[X[:, 0] <= X[:, 1]] = 0

# Flip the labels make the data non-separable
for i in range(0, len(Y)):
    r = np.random.uniform(0, 1)
    if r <= 0.1:
        if Y[i] == 0:
            Y[i] = 1
        else:
            Y[i] = 0

# Name of the column
Synthetic_df_X = pd.DataFrame(X, columns=['First_element', 'Second_element'])

# Figure of the data
fig = plt.figure()
ax = plt.axes()
B_data = Synthetic_df_X['First_element']
G_data = Synthetic_df_X['Second_element']
ax.scatter(B_data, G_data, c=Y, marker=".")
plt.show()
