# RESAMPLING 
# Split training data into subsets;
# some used to train, others used to validate

#%%
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'
data = pd.read_csv(url, delimiter=',')
values = data.values # Gets array of data values
X = values[:, 0:11] # Gets input values
Y = values[:, 11] # Gets output values (quality)

kfold = KFold(n_splits=10, random_state=7) # Splits data into "k consecutive folds"
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
mean = round(results.mean() * 100, 2)
std = round(results.std() * 2, 2)
print(f'Accuracy: {mean} (+/- {std})')