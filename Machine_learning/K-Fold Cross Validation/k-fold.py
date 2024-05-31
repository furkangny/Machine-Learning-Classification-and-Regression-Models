# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:14:51 2024

@author: frkng
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# %%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization
x = (x-np.min(x))/(np.max(x)-np.min(x))

# %% train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

# %% KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

# %% 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn, X= x_train, y= y_train, cv = 10)
print("average accuracy :", np.mean(accuracies))

#%%
knn.fit(x_train, y_train)
print("test accuracy:", knn.score(x_test,y_test))

# %% grid search cross validation
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10) # GridSearch cv
knn_cv.fit(x,y)

# %% print HyperParameter KNN alhoritmasındaki k değeri

print("tuned hyperparameter K:", knn_cv.best_params_)
print("best score: ", knn_cv.best_score_)

# %% Grid Search CV with logistic regression
x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3, 3, 7), "penalty":["l1","l2"]} # l1 = lasso , l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameter: (best parameters):", logreg_cv.best_params_)



























