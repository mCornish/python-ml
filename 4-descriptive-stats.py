import pandas as pd

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'
data = pd.read_csv(url, delimiter=',')
print(data.head())
print('Shape: ', data.shape)
print(data.dtypes)
print(data.describe())

# Pearson correlation - 0 => no relationship, -1 => perfect inverse linear rel., 1 => perfect linear rel.
print(data.corr())