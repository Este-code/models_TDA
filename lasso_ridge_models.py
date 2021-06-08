import numpy as np
import pandas as pd
from scipy.stats.stats import mode
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

data = pd.read_csv('uni_data.csv') # load our data

x = data.iloc[:-1, 1:9] # set predicting varaibles
y = data.iloc[:-1, 9] # set varaible to predict (score)

# define models
model_lasso = Lasso(alpha=0.01)
model_ridge = Ridge(alpha=0.01)
# fit models
model_lasso.fit(x,y)
model_ridge.fit(x,y)

# define new data to predict (selecting one university)
new = data.iloc[1111, 1:9]

#make predictions
prediciton_lasso = model_lasso.predict([new])
prediction_ridge = model_ridge.predict([new])

# summarize predictions 
print("Predicted: %.3f" % prediciton_lasso)
print("Predicted: %.3f" % prediction_ridge)