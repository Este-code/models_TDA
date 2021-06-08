import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from math import sqrt

data = pd.read_csv('fruit_types.csv')

encoder = preprocessing.LabelEncoder()

fruit_name_encoded = encoder.fit_transform(data.iloc[:,0])
fruit_subtype_encoded = encoder.fit_transform(data.iloc[:,1])
mass_encoded = encoder.fit_transform(data.iloc[:,2])
width_encoded = encoder.fit_transform(data.iloc[:,4])
height_encoded = encoder.fit_transform(data.iloc[:,5])

data = list(zip(fruit_name_encoded,fruit_subtype_encoded,mass_encoded,width_encoded,height_encoded))
data = pd.DataFrame(data, columns = ['fruit_name','fruit_subtype','mass','width','height'])

data_to_use = data.iloc[:,1:5]
data_to_target = data.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(data_to_use, data_to_target, test_size = 0.2, random_state = 1)

knn_model = KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights='distance')

knn_model = knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print("RMSE: ", sqrt(metrics.mean_squared_error(y_test,y_pred)))
