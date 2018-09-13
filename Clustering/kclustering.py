# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:33:56 2018

@author: saket
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[ : , [3 , 4]].values

#using elbow method to find max number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++' , random_state =0 , max_iter= 300 , n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)

#applying kmeans to optimal cluster number
kmeans = KMeans(n_clusters =5, init ='k-means++' , random_state = 0 , max_iter=300, n_init = 10)
ykmeans = kmeans.fit_predict(X)

#plotting clusters indifferent colors
plt.scatter(X[ykmeans==0, 0], X[ykmeans == 0 , 1], s= 100 , c = 'red' , label ='cluster 1')  
plt.scatter(X[ykmeans==1, 0], X[ykmeans == 1 , 1], s= 100 , c = 'green' , label ='cluster 2')  
plt.scatter(X[ykmeans==2, 0], X[ykmeans == 2 , 1], s= 100 , c = 'blue' , label ='cluster 3')  
plt.scatter(X[ykmeans==3, 0], X[ykmeans == 3 , 1], s= 100 , c = 'orange' , label ='cluster 4')  
plt.scatter(X[ykmeans==4, 0], X[ykmeans == 4, 1], s= 100 , c = 'pink' , label ='cluster 5')
plt.scatter(kmeans.cluster_centers_[: ,0], kmeans.cluster_centers_[: ,1], s=300, c= 'black')
  
