# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:34:59 2024

@author: frkng
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv", sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("araba max hiz")
plt.xlabel("araba fiyat")
plt.show()

# linear reg -> y = b0 + b1*x
#multiple linear reg -> y = b0 + b1*x1 + b2*x2

#%%

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict

y_head = lr.predict(x)

plt.plot(x,y_head, color = "red", label = "linear")
plt.show()

# göründüğü üzere bu dataset linear reg için uygun değil

#%%

from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree = 3)
x_polynomial = polynomial_reg.fit_transform(x)

#%% fit

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial, y)

#%%visualize

y_head2 = linear_reg2.predict(x_polynomial)

plt.plot(x,y_head2, color = "green", label = "poly")
plt.legend()
plt.show()




































