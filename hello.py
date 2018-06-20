import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Matplotlib
x = np.arange(0, 1, .03)
for n in [1, 2, 3, 4]:
  plt.plot(x, x**n, label='n=%d'%(n,))

leg = plt.legend(loc='best', ncol=2, mode='expand', shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

# plt.show()


# Pandas
s = pd.Series(np.arange(0, 10, 2))
# print(s)

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('xxji'))
# print(df)

df2 = pd.DataFrame({
  'A': 1.,
  'B': pd.Timestamp('2018/03/14'),
  'C': pd.Series(1, index=list(range(4)), dtype='int16'),
  'D': np.array([3] * 4),
  'E': pd.Categorical(['bounce', 'bounce', 'roll', 'bounce']),
  'F': 'bar'
})
print(df2, df2.dtypes)