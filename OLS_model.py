import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('car_data.csv') # load our data

# select the 2nd and 3rd columns and convert them to a numpy array
x = data.iloc[:, 2].values.reshape(-1, 1)
y = data.iloc[:, 3].values.reshape(-1, 1)


linear_regression = LinearRegression() # create a linear regression object
linear_regression.fit(x,y) # perform the linear regression
Y_pred = linear_regression.predict(x) # make the prediction


# plotting data a predicted variable
plt.scatter(x, y)
plt.plot(x, Y_pred, color='red')
plt.xlabel('Selling_price')
plt.ylabel('Km driven')
plt.show()
