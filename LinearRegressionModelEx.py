import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df=pd.read_csv('USA_Housing.csv')
print(df.head())
#sns.pairplot(df)
#plt.tight_layout()
#plt.show()
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',                           #Features
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']                                                                                                                                                               #Labels
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
print('X_TRAIN VALUES')
print(X_train.head())
print('X_TEST VALUES')
print(X_test.head())
print('Y_TRAIN VALUES')
print(y_train.head())
print('Y_TEST VALUES')
print(y_test.head())
lm=LinearRegression()
lm.fit(X_train,y_train)  #fitting training values
#print(lm.intercept_)
#print(lm.coef_)
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['coef'])  #Coefficient DataFrame   (data,index,columns)
#print(cdf)
print('PREDICTIONS')
pred=lm.predict(X_test)
print(pred)
print('MEAN_ABSOLUTE_ERROR')
print(metrics.mean_absolute_error(y_test,pred))
#plt.scatter(pred,y_test)
sns.distplot((y_test-pred))    #plot the residuals in correct manner
plt.xlabel('PREDICTIONS')
plt.ylabel('Y_TEST')
plt.tight_layout()
plt.show()
