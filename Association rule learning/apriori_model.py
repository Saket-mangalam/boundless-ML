# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:42:57 2018

@author: saket
"""

#import class
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction =[]
for i in range (0,7501):
    transaction.append([str(dataset.values[i, j]) for j in range (0,20)])

#train on apyori file
from apyori import apriori
rules =apriori(transaction , min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualisation of results
result = list(rules)