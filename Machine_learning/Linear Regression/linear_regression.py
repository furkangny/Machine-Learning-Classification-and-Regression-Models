# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:36:09 2024

@author: frkng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")

#%%

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%%
print(linear_reg.predict([[11]]))
 

#visualize line

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color = "red")















