# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:09:38 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",rf.predict([[7.8]]))

y_head = rf.predict(x)

#%%

from sklearn.metrics import r2_score

print("r score: ", r2_score(y, y_head))



















