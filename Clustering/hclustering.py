# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:48:59 2018

@author: saket
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: ,[3,4]].values

#using dendogram to find optimal clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.show()

#fitting h clustering
from sklearn.cluster import AgglomerativeClustering
hclust = AgglomerativeClustering(n_clusters= 5, affinity = 'euclidean', linkage ='ward')
y = hclust.fit_predict(X)

#plotting results
plt.scatter(X[y==0, 0], X[y == 0 , 1], s= 100 , c = 'red' , label ='cluster 1')  
plt.scatter(X[y==1, 0], X[y == 1 , 1], s= 100 , c = 'green' , label ='cluster 2')  
plt.scatter(X[y==2, 0], X[y == 2 , 1], s= 100 , c = 'blue' , label ='cluster 3')  
plt.scatter(X[y==3, 0], X[y == 3 , 1], s= 100 , c = 'orange' , label ='cluster 4')  
plt.scatter(X[y==4, 0], X[y == 4, 1], s= 100 , c = 'pink' , label ='cluster 5')

  
