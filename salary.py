import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		
df=pd.read_csv(r"C:\Users\sss\Downloads\Salary_Data.csv")
x=df.iloc[:,:-1]
y=df.iloc[: ,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

m=regressor.coef_
c=regressor.intercept_
(m*12)+c
(m*20)+c
#mean

df.mean() 
df['Salary'].mean() 
#median

df.median()
df['Salary'].median()
#mode

df['Salary'].mode()
#varience

 df.var()
 df['Salary'].var()
 #standard deviation
 
 df.std()
 df['Salary'].std() 
 # Coefficient of variation(cv)
 
 from scipy.stats import variation
 variation(df.values)
 variation(df['Salary'])
 
 # Correlation
 
 df.corr()
 df['Salary'].corr(df['YearsExperience']) 
 #Skewness
 
 df.skew() 
 df['Salary'].skew()
 #Standard Error
 
 df.sem() 
 df['Salary'].sem()
 #z-score
 
 import scipy.stats as stats
 df.apply(stats.zscore) 
 stats.zscore(df['Salary']) 
 a = df.shape[0] # this will gives us no.of rows
 b = df.shape[1] # this will give us no.of columns
 degree_of_freedom = a-b
 print(degree_of_freedom)
 # SSR
 
 SR = np.sum((y_predict-y_mean)**2)
 print(SSR)
 #SSE
 
 y = y[0:6]
 SSE = np.sum((y-y_predict)**2)
 print(SSE)
  #SST
  
  mean_total = np.mean(df.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((df.values-mean_total)**2)
print(SST)

 #r-Square
 
 r_square = SSR/SST
r_square
 
 
 
 
 plt.scatter(x_test,y_test,color="red")
 plt.plot(x_train,regressor.predict(x_train),color='blue')
 plt.title('salary vs Experience (test data)')
 plt.xlabel('years of Experience')
 plt.ylabel('salary')
 plt.show()
 
 
 