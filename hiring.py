
import pandas as pd
import numpy as np

df=pd.read_csv('/content/hiring.csv')
df

df.head

import math

test_median=math.floor(df['test_score(out of 10)'].median())
test_median

df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(test_median)
df

df.experience=df.experience.fillna("zero")

df

df.dtypes

pip install word2number

from word2number import w2n

df.experience=df.experience.apply(w2n.word_to_num)
df

df['experience'].dtypes

import matplotlib.pyplot as plt

plt.plot(df.experience,df['salary($)'])

from sklearn import linear_model

reg=linear_model.LinearRegression()

reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

reg.predict([[2,9,6]])

reg.predict([[12,10,10]])

