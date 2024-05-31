# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:16:20 2024

@author: frkng
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep = ";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

#%%

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

print(multiple_linear_reg.predict([[10,35],[5,35]]))