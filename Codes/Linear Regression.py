# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import data
Data = pd.read_csv('Dummy Data HSS.csv', header=None)

# Delete the first row because they are name
Data = Data.iloc[1:, :]

# Rename the columns
column_names = {0: 'TV', 1: 'Radio', 2: 'Social_Media', 3: 'Influencer', 4: 'Sales'}
Data.rename(columns=column_names, inplace=True)

# Drop nan rows in Radio column
Data.dropna(subset=["Radio"], inplace=True)

# Drop nan rows in Sales column
Data.dropna(subset=["Sales"], inplace=True)

# Separate X and Y
predictors = ['Radio']
outcome = ['Sales']
X = Data[predictors]
Y = Data[outcome]

# train_test_split 80% data for training and keep 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Linear Regression Model with Regularization
linReg = LinearRegression()
model_lin = linReg.fit(X_train, Y_train)
y_prediction_lin = linReg.predict(X_test)

# Figure Radio advertisement cost and sales for linear regression
a1 = X_test['Radio'].tolist()
a2 = Y_test['Sales'].tolist()
Radio = [float(x) for x in a1]
Sales = [float(x) for x in a2]
m, b = np.polyfit(Radio, Sales, 1)
plt.plot(Radio, Sales, '.')
Radio_Y = [x * m + b for x in Radio]
plt.plot(Radio, Radio_Y)
plt.title("Linear regression on sales based on Radio advertisement")
plt.xlabel('Radio promotion budget (in million)')
plt.ylabel('Sales (in million)')
plt.show()
