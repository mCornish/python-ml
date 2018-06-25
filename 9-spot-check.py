# SPOT-CHECK ALGORITHMS
# Discover best-performing algorithm

#%%
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
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

# Prepare models
models = [
  ('LGR', LogisticRegression()),
  ('GNB', GaussianNB()),
  ('SVM', SVC(kernel='linear')),
  ('KNN', KNeighborsClassifier()),
  ('DTC', DecisionTreeClassifier(min_samples_split=2500)),
  ('DTR', DecisionTreeRegressor()),
  ('RFC', RandomForestClassifier()),
  ('GBC', GradientBoostingClassifier())
]

predictions = []
# Evaluate models
for name, model in models:
  clf = model.fit(X_train, Y_train)
  pred = clf.predict(X_test)
  predictions.append(pred)
  print(f'{name} Accuracy: ', accuracy_score(Y_test, pred))

#%%
for prediction in predictions:
  plt.plot(X_test, prediction)
