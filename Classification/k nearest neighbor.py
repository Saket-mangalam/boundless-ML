# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:01:06 2018

@author: saket
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2 , 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#logistic regression is linear model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski' , p =2)
classifier.fit(X_train, y_train)

#predicting results
y_pred = classifier.predict(X_test)

#finding the prediction smartness
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plot for logistic regression
#fucking try and learn how to do this. If you get this, you are ready for company
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[:, 0].max()+1, step =0.01),
                              np.arange(start = X_set[: , 1].min()-1 , stop = X_set[: , 1].max()+1 , step=0.1))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75 , cmap =ListedColormap(('red','green')) )
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],X_set[y_set == j, 1], c = ListedColormap(('red','green'))(i), label =j)
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[:, 0].max()+1, step =0.01),
                              np.arange(start = X_set[: , 1].min()-1 , stop = X_set[: , 1].max()+1 , step=0.1))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75 , cmap =ListedColormap(('red','green')) )
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],X_set[y_set == j, 1], c = ListedColormap(('red','green'))(i), label =j)
plt.legend()
plt.show()

                              
