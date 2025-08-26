import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


#load the dataset
dataset = pd.read_csv(r"C:\Users\sss\Downloads\20th - mlr\MLR\House_data.csv")

# Split the data independnet into dependent variable
x = dataset[['sqft_living']]
y = dataset['price']

#Split the dataset into training and testing sets(80-20%)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#trai the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predict test set
y_pred = regressor.predict(x_test)

#Visualize trsining set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Space vs price (Training set)')
plt.xlabel('space')
plt.ylabel('price')
plt.show()

#VIsualize test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Space vs price (Test set)')
plt.xlabel('space')
plt.ylabel('price')
plt.show()


y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted house price after 12 years: ${y_12[0]:,.2f}")
print(f"Predicted house price after 20 years : ${y_20[0]:,.2f}")


#check model performance
bias = regressor.score(x_train,y_train)
variance = regressor.score(x_test,y_test)
train_mse = mean_squared_error(y_train,regressor.predict(x_train))
test_mse = mean_squared_error(y_test,y_pred)

print(f"Training Score (R^2):{bias:.2f}")
print(f"Testing Score (R^2):{variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE : {test_mse:.2f}")


#save trained mode to disk
filename = 'linear_regression_housemodel.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("model has been pickled and saved as linear_regression_housemodel.pkl")
import os
os.getcwd()