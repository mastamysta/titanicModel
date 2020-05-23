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
#assigning mean value to nan age vals
for i in range(Len):
    if math.isnan(ageNorm[i]):
        ageNorm[i] = ageMean
        
cabinNorm = [0] * Len


#fidning classes of cabins
for i in range(Len):
    if isinstance(trainData["Cabin"][i], str):
        if trainData["Cabin"][i][0] == "A":
            cabinNorm[i] = 0
        elif trainData["Cabin"][i][0] == "B":
            cabinNorm[i] = 1
        elif trainData["Cabin"][i][0] == "C":
            cabinNorm[i] = 2
        elif trainData["Cabin"][i][0] == "D":
            cabinNorm[i] = 3
        elif trainData["Cabin"][i][0] == "E":
            cabinNorm[i] = 4
        elif trainData["Cabin"][i][0] == "F":
            cabinNorm[i] = 5
        elif trainData["Cabin"][i][0] == "G":
            cabinNorm[i] = 6
        else:
            cabinNorm[i] = 7
    else:
        if math.isnan(trainData["Cabin"][i]):
            cabinNorm[i] = 8
            

#normalise embarked values
embarkedNorm = [0] * Len
for i in range(Len):
    if not isinstance(trainData["Embarked"][i], str):
        embarkedNorm[i] = 0
    else:
        if trainData["Embarked"][i] == "S":
            embarkedNorm[i] = 1
        elif trainData["Embarked"][i] == "C":
            embarkedNorm[i] = 2
        elif trainData["Embarked"][i] == "Q":
            embarkedNorm[i] = 3


X = pd.DataFrame()
X["Pclass"] = trainData["Pclass"]
X["Sex"] = genderNorm
X["Age"] = ageNorm
X["SibSp"] = trainData["SibSp"]
X["Parch"] = trainData["Parch"]
#WILL RETURN TO TICKET NUMBERS
#X["Ticket"] = trainData["Ticket"]
X["Fare"] = trainData["Fare"]
X["Cabin"] = cabinNorm
X["Embarked"] = embarkedNorm


model = DecisionTreeRegressor(random_state=1)
model.fit(X, Y)

print(features)