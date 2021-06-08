import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

data = pd.read_csv('uni_data.csv') # load our data

x = data.iloc[:-1, 1:9] # set predicting varaibles
y = data.iloc[:-1, 9] # set varaible to predict (score)

# define model evaluation method
CV = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

# define model
model = RidgeCV(alphas=np.arange(0, 1, 0.1), cv=CV, scoring='neg_mean_absolute_error')

# fit model
model.fit(x, y)

print('alpha: %f' % model.alpha_)