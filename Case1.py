# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:04:14 2019

@author: sabya
"""

# =============================================================================
# First Liner Regression Case Studies
# =============================================================================
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
# =============================================================================
# Importing the dataset
# =============================================================================
os.chdir('C:/Users/sabya/OneDrive/Desktop/ivy/python ml/multiple linear/case study')
path_data = os.getcwd()
mydata = pd.read_csv('weatherHistory.csv')
mydata.head(5)
# =============================================================================
# Information of the variables
# =============================================================================
mydata.info()
#Categorical variables:
categorical = mydata.select_dtypes(include = ["object"]).keys()
print(categorical)

#Quantitative variables:
quantitative = mydata.select_dtypes(include = ["int64","float64"]).keys()
print(quantitative)

#Quantitative variables. Missing values
mydata[quantitative].describe()
a=mydata[quantitative].describe()
a.info()
# =============================================================================
#Detecting and removing outliers
# =============================================================================
sns.boxplot(x=mydata['Humidity'])
#Removing the Outliers
mydata1=mydata[mydata["Humidity"]>0.2]
sns.boxplot(x=mydata['Temperature (C)'])
mydata2=mydata1[mydata1["Temperature (C)"]>-15]
mydata1=mydata2
# #Missing Value
mydata1.isnull().sum()
# =============================================================================
# First drop 517 missing value from precip type
# =============================================================================
mydata1.dropna(inplace=True)
# plotting a scatter plot between temp and humdity
Cor=mydata1.corr()
data_set=mydata1.iloc[:,[0,2,3,4,5,8]]
Cor_s=data_set.corr()
sns.regplot(x=data_set["Temperature (C)"], y=data_set["Humidity"])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# =============================================================================
# Creating the Independendent and Dependent Data Sets
# =============================================================================
B = mydata1.iloc[:,[1,2,4,5,6,7,8,10]]  #independent columns
C = mydata1.iloc[:,[3]]    #target column i.e temparature

# =============================================================================
# Convert catagorical data to labelenconder
# =============================================================================
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
B['Summary'] = labelencoder.fit_transform(B['Summary'])
X1 = pd.DataFrame(B)

X1 = X1.rename(columns = {"Precip Type":"Precip_Type"}) 

labelencoder = LabelEncoder()
X1['Precip_Type'] = labelencoder.fit_transform(X1['Precip_Type'])
X2 = pd.DataFrame(X1)


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, C, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_data=pd.DataFrame(y_pred)

regressor.score(X_train,y_train)
regressor.score(X_test,y_test)

#Root Mean Square Error
from sklearn import metrics
import math
print(math.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# =============================================================================
# #Model Statistics
# =============================================================================

#Adding Intercept term to the model
X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)

#Converting into Dataframe
X_train_d=pd.DataFrame(X_train)

#Printing the Model Statistics
model = sm.OLS(y_pred,X_test).fit()
model.summary()

#Checking the VIF Values
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] =[variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]
vif["features"] = X_train_d.columns
vif.round(1)

New_X_train=np.array(X_train.drop(["Apparent Temperature (C)"],axis = 1,inplace = True))
New_X_test=np.array(X_test.drop(["Apparent Temperature (C)"],axis = 1,inplace = True))

regressor.fit(New_X_train, y_train)



model1 = sm.OLS(y_pred,New_X_test).fit()
model1.summary()
