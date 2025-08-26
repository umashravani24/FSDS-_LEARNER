import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		
#load the dataset
dataset = pd.read_csv(r"C:\Users\sss\Downloads\20th - mlr\MLR\House_data.csv")

# Split the data independnet into dependent variable
x = dataset[['sqft_living','bedrooms','bathrooms']]
y = dataset['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 
x = np.append(arr=np.full((21613,1), 67512).astype(int), values=x, axis=1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

bias = regressor.score(x_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance



