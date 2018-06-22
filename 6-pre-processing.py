#%%
import pandas as pd
from sklearn import preprocessing
import numpy as np

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'
data = pd.read_csv(url, delimiter=',')
values = data.values # Gets array of data values
X = values[:, 0:11] # Gets input values
Y = values[:, 11] # Gets output values (quality)

# STANDARDIZE (e.g. mean = 0, stan. dev. = 1)
X_scaled = preprocessing.scale(X)
print(X_scaled.mean(axis=0), X_scaled.std(axis=0))

#%%
#NORMALIZE (e.g. range 0 - 1)
normalizer = preprocessing.Normalizer()
X_normalized = normalizer.transform(X)
np.set_printoptions(precision=3)
print(X)
print(X_normalized)

#%%
#BINARIZE (e.g. boolean values)
binarizer = preprocessing.Binarizer()
X_binary = binarizer.transform(X)
print(X_binary)