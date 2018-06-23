# ALORITHM EVALUATION METRICS
# Evaluate skill of an algorithm on a dataset

#%%
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'
data = pd.read_csv(url, delimiter=',')
values = data.values # Gets array of data values
X = values[:, 0:11] # Gets input values
Y = values[:, 11] # Gets output values (quality)

kfold = KFold(n_splits=10, random_state=7) # Splits data into "k consecutive folds"
for train_index, test_index in kfold.split(X):
  X_train, X_test = X[train_index], X[test_index]
  Y_train, Y_test = Y[train_index], Y[test_index]

model = LogisticRegression()
model = model.fit(X_train, Y_train)

prediction = model.predict(X_test)

print('Accuracy: ', accuracy_score(Y_test, prediction))
print('MSE: ', mean_squared_error(Y_test, prediction))
print('R2: ', r2_score(Y_test, prediction))
print()
print(confusion_matrix(Y_test, prediction))

#%%
for index, val in enumerate(X):
  plt.scatter(X[:, index], Y)
  plt.plot(X_test, prediction)