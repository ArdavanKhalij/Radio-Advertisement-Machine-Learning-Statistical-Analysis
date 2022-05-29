# Libraries
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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
    if r <= 0.01:
        if Y[i] == 0:
            Y[i] = 1
        else:
            Y[i] = 0

# Name of the columns
Synthetic_df_X = pd.DataFrame(X, columns=['First column', 'Second column'])
Synthetic_df_Y = pd.DataFrame(Y, columns=['Result'])

# train_test_split 80% data for training and keep 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(Synthetic_df_X, Synthetic_df_Y, test_size=0.2, random_state=0)

# Naive Bayes classification
nb = GaussianNB()
nb.fit(X_train, Y_train)
y_prediction_Naive_Bayes = nb.predict(X_test)

# Decision Tree classification
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, Y_train)
y_prediction_Decision_Tree = dt.predict(X_test)

# Figure of the data
fig = plt.figure()
ax = plt.axes()
First_element = Synthetic_df_X['First column']
Second_element = Synthetic_df_X['Second column']
Result = Synthetic_df_Y['Result']
ax.scatter(First_element, Second_element, c=Result, marker=".")
plt.title("Synthetic data")
plt.xlabel('First column')
plt.ylabel('Second column')
plt.show()
