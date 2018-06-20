import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'
data = pd.read_csv(url, delimiter=',')

# data['alcohol'].hist(orientation='horizontal')
# data.loc[:, ['volatile acidity', 'citric acid']].plot(kind='box')
pd.plotting.scatter_matrix(data.iloc[:, 0:5])

print(data.head())
plt.show()
