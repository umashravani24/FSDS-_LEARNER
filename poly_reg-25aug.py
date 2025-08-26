import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline




dataset = pd.read_csv(r"C:/Users/sss/Downloads/emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Build the model

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(x,y)

#linear regression visualization

plt.scatter(x,y,color='black')
plt.plot(x,lin_reg.predict(x),color='red')
plt.title('Linear regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

linear_model_pred=lin_reg.predict([[6.5]])    

print(linear_model_pred) 

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=5)

x_poly=poly_reg.fit_transform(x)   

poly_reg.fit(x_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x, y, color='green')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(Polynomial features)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()       

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

print(poly_model_pred)
