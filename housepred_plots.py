import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		
dataset=pd.read_csv(r"C:\Users\sss\Downloads\15th- SLR\SLR - House price prediction\House_data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[: ,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#Visualizing the training Test Results 
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Visuals for Train DataSet)')
plt.xlabel('space')
plt.ylabel('price')
plt.show()

#Visualizing the Test Results 
plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
m=regressor.coef_
c=regressor.intercept_
(m*12)+c
(m*20)+c
