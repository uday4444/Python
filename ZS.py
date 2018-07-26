# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:51:33 2018

@author: ASUS
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
train=pd.read_csv('yds_train2018.csv')
print(train.head())
test=pd.read_csv('yds_test2018.csv')
print(test.head())
#sns.heatmap(test.isnull(),cbar=False,cmap='viridis',yticklabels=False)
#print(train.isnull())
#print(pd.get_dummies(train['Country']))
#train=train[['S_No','Year','Month','Product_ID','Sales']]
#print(train.head())
#test=test[['S_No','Year','Month','Product_ID','Sales']]
#sns.heatmap(test.isnull(),cbar=False,cmap='viridis',yticklabels=False)
#print(test.head())
#sns.heatmap(train.isnull(),cbar=False,cmap='viridis',yticklabels=False)
#print(train.head())
#print(train.columns)
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#ttc=pd.concat([train,test],axis=0)
#print(ttc)
#Salesavg=ttc['Sales'].mean()
#print(Salesavg)
#def avg(cols):
#    if pd.isnull(cols):
#        cols=Salesavg
#        return cols
#    else:
#        cols=cols
#        return cols
#ttc['Sales']=ttc['Sales'].apply(avg)
#print(ttc)
X=train.drop(['Country','Sales','Merchant_ID'],axis=1)
y=train['Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(pred)
#preddf=pd.DataFrame(pred)
#plt.xlim((0,10))
#sns.distplot(y_test-pred)
print(r2_score(y_test,pred))
plt.scatter(y_test,pred)
plt.xlabel('y_test')
plt.ylabel('predictions')