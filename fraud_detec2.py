import numpy as np
import pandas as pd
from pandas.core.indexes.period import PeriodIndex
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('fraud_detection.csv') # load our data

data.index = data.iloc[:,0]
data_to_use = data.iloc[:,[2,4,5,7,8]]
data_to_target = data.iloc[:, 9]

x_train, x_test, y_train, y_test = train_test_split(data_to_use, data_to_target, test_size=0.4, random_state=123)

gaussian_model = GaussianNB()
gaussian_model.fit(x_train,y_train)

prediction = gaussian_model.predict(x_test)

result = {'isFraud':y_test, 'pred':prediction}
result_df = pd.DataFrame(result)
print(result_df)
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))