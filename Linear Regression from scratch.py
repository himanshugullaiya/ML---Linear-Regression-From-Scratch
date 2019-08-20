import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size = 1/3, random_state = 0)

# y = mx + c
alpha = 0.001 
c = 0
m = 0
sample_size = len(X_train)

def der(m , c):
      global X_train, Y_train, alpha, sample_size
      der_c , der_m = 0,0
      
      for i in range(sample_size):
            der_m += X_train[i]*((m*X_train[i] + c) - Y_train[i])
            der_c += ((m*X_train[i] + c) - Y_train[i])
      
      der_m /= sample_size;
      der_c /= sample_size;
      
      return (alpha*der_m , alpha*der_c)

count = 0
d_m = der(m,c)[0]
temp_c , temp_m = 0,0
while  d_m <= 0:
      d_c = der(m,c)[1]
      d_m = der(m,c)[0]
      
      temp_c = c - d_c
      temp_m = m - d_m
      
      m = temp_m
      c = temp_c
      count += 1

z = [m*i+c for i in  X_train]

z = np.asarray(z)


# plotting the graph
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1) , Y_train.reshape(-1,1))
Y_reg = regressor.predict(X_train.reshape(-1,1))


##################################################

#Plotting the model

plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, z, color = "blue")      
plt.plot(X_train.reshape(-1,1), Y_reg , color = "green")
  

# r2 score

from sklearn.metrics import r2_score

print(" My Model Accuracy: ",r2_score(Y_train,z))
print(" Sklearn Accuracy:  ",r2_score(Y_train,Y_reg))


            
      
      
      
      
                 