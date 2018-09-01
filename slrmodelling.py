# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:44:01 2018

@author: saket
"""

import numpy as np #helps deal with numeric values
import matplotlib.pyplot as plt # for plots and graphs
import pandas as pd #for importing datasets

#importing the dataset
dataset= pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 1].values

#missing data management
"""from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy= 'mean' , axis = 0)
imputer= imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])"""
 
#data encoding from sklearn.preprocessing import LabelEncoder, OneHotEncoder
"""labelencoder_x=LabelEncoder()
X[:, 0]=labelencoder_x.fit_transform(X[:, 0])
onehotencoder_x= OneHotEncoder(categorical_features=[0])
X = onehotencoder_x.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""

#train test splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
standardscalerX = StandardScaler()
X_train = standardscalerX.fit_transform(X_train)
X_test = standardscalerX.transform(X_test)"""

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_predict = regressor.predict(X_test)

#plotting predictions
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('salary vs experience')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

#plotting test predictions
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('salary vs experience')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

