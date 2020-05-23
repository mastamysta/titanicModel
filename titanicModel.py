#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:18:51 2020

@author: ben
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import math

trainData = pd.read_csv("data/train.csv")
#print(trainData.describe())
#print(trainData.head())
features=trainData.columns[2:]
features = features.drop("Name")
label=trainData.columns[1]
#print(features)
#print(label)
#X = trainData[features]
Y = trainData[label]



#normalise gender data
gender = trainData["Sex"]
Len = len(gender)
genderNorm = [0] * Len
for i in range(Len):
    if gender[i] == "male":
        genderNorm[i] = True
    else:
        genderNorm[i] = False

ageNorm = trainData["Age"]

#finding mean age of pressent data
ageMean = 0
count = 0
summation = 0
for i in range(Len):
    if not math.isnan(ageNorm[i]):
        count += 1
        summation += ageNorm[i]
ageMean = summation / count
for i in range(Len):
    if math.isnan(ageNorm[i]):
        ageNorm[i] = ageMean

X = pd.DataFrame()
X["Pclass"] = trainData["Pclass"]
X["Sex"] = genderNorm
X["Age"] = ageNorm

model = DecisionTreeRegressor(random_state=1)
model.fit(X, Y)


