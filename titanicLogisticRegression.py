# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:50:07 2018

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
train=pd.read_csv('titanic_train.csv')
#print(train.head())
#print(train.isnull())
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',inplace=True,axis=1)
#print(train.head())
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.boxplot(x='Pclass',y='Age',data=train)
def funct(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return  37                        #DATASET CLEANING ....AGE COLUMN
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(funct,axis=1)
#train.drop(['Name','Sex','Ticket','Embarked'],inplace=True,axis=1)
u=pd.get_dummies(train['Sex']).drop(['female'],axis=1)
train['Sex']=u
train.drop(['Name','Ticket','Embarked'],inplace=True,axis=1)   
print(train.head())
X=train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
y=train['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y)
#print(X_train)                                #TRAINING VALUES
log=LogisticRegression()
n=log.fit(X_train,y_train)                     #FITTING TRAINING VALUES 
print(n)
pred=log.predict(X_test)                       #PREDICTION 
print(pred)
#sns.barplot(y_test,pred)
print(classification_report(y_test,pred))       #PRECISION 
