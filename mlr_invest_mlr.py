import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
 
	
dataset=pd.read_csv(r"C:/Users/sss/Downloads\Investment.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#== we build mlr model 

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 
X = np.append(arr=np.full((50,1), 42467).astype(int), values=X, axis=1)


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
#Visualize trsining set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()

#VIsualize test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")


#check model performance
bias = regressor.score(X_train,y_train)
variance = regressor.score(X_test,y_test)
train_mse = mean_squared_error(y_train,regressor.predict(X_train))
test_mse = mean_squared_error(y_test,y_pred)

print(f"Training Score (R^2):{bias:.2f}")
print(f"Testing Score (R^2):{variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE : {test_mse:.2f}")


#save trained mode to disk
filename = 'multilinear_regression_investmentmodel.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("model has been pickled and saved as multilinear_regression_investmentmodel.pkl")
import os
os.getcwd()





