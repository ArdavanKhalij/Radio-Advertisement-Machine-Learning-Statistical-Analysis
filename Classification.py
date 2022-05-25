# Libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib

# Convert txt to csv
# column_names = ['B', 'G', 'R', 'Class']
# read_file = pd.read_csv(r'Skin_NonSkin.txt', delimiter='\t')
# read_file.to_csv(r'Skin_NonSkin.csv', header=None)

# Load the csv data and give column names
column_names = ['B', 'G', 'R', 'Class']
Data = pd.read_csv('Skin_NonSkin.csv', names=column_names, header=None)

# Naive Bayes classification
nb = GaussianNB()
excludeColumns = 'Class'
predictors = [s for s in Data.columns if s not in excludeColumns]
outcome = ['Class']
X = Data[predictors]
Y = Data[outcome]
nb.fit(X, Y)
