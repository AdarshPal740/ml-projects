
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/india gdp.csv')
df

df.dtypes

df.shape

df.describe()

hgt = data.Height.values.reshape(-1,1)

df['DATE'] = pd.to_datetime(df['DATE'])
df['year'] = df['DATE'].dt.year
df
df.plot(kind='scatter', x='year', y='NYGDPPCAPKDIND')

x = df.loc[:,'year'].values.reshape(-1,1)
y = df.loc[:,'NYGDPPCAPKDIND'].values.reshape(-1,1)



df.plot(kind='bar', x='year', y='NYGDPPCAPKDIND')

df.dtypes

df.tail

plt.plot(df['year'], df['NYGDPPCAPKDIND'],color="red")
plt.xlabel("year")
plt.ylabel("Per capita income")
# plt.legend(loc="upper")

plt.hist(df['NYGDPPCAPKDIND'])
plt.show()



plt.hist(df['DATE'])
plt.show()

from sklearn import linear_model

plt.scatter(df.DATE,df.NYGDPPCAPKDIND)

reg=linear_model.LinearRegression()
reg.fit(x,y)

reg.predict([[2001]])

score=reg.score(x,y)
print(score)

import statsmodels.api as sl
model=sl.OLS(y,x)
model=model.fit()
model.summary()

model.predict(2001)

reg.coef_

reg.intercept_

24.13344777*2001+-47295.9468294

