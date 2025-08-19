import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		
dataset=pd.read_csv(r"C:\Users\sss\Downloads\Salary_Data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[: ,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('price vs location (test data)')
plt.xlabel('total_sqft')
plt.ylabel('price')
plt.show()
m=regressor.coef_
c=regressor.intercept_
(m*12)+c
(m*20)+c