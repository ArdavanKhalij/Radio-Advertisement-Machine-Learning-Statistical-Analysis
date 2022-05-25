# Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert txt to csv
# column_names = ['B', 'G', 'R', 'Class']
# read_file = pd.read_csv(r'Skin_NonSkin.txt', delimiter='\t')
# read_file.to_csv(r'Skin_NonSkin.csv', header=None)

# Load the csv data and give column names
column_names = ['B', 'G', 'R', 'Class']
Data = pd.read_csv('Skin_NonSkin.csv', names=column_names, header=None)

# Separate predictors and outcome
excludeColumns = 'Class'
predictors = [s for s in Data.columns if s not in excludeColumns]
outcome = ['Class']
X = Data[predictors]
Y = Data[outcome]

# Balance the data
under_sample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, Y_train_under = under_sample.fit_resample(X, Y)

print(Y_train_under.value_counts())

# Naive Bayes classification
nb = GaussianNB()
nb.fit(X_train_under, Y_train_under)
# y_prediction = nb.predict(X_test)

# Naive Bayes classification
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train_under, Y_train_under)
# y_prediction = dt.predict(X_test)

# Figure for the points in the dataset
fig = plt.figure()
ax = plt.axes(projection='3d')
B_data = X_train_under['B']
G_data = X_train_under['G']
R_data = X_train_under['R']
Class_data = Y_train_under['Class']
ax.scatter3D(B_data, G_data, R_data, c=Class_data, marker=".")
plt.show()
