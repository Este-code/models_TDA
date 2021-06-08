import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm

# Prepare the data
data = pd.read_csv('fruit_types.csv')
X = data.iloc[:,2:5]
Y = data.iloc[:,0]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1)

#Create 3 SVM Classifiers
linear_SVM = svm.SVC(kernel='linear', gamma ='auto') # Linear Kernel
sigmoid_SVM = svm.SVC(kernel='sigmoid', shrinking=False) # Sigmoid Kernel
rbf_SVM = svm.SVC(kernel='rbf', gamma ='auto') # RBF Kernel

#Train the model using the training sets
linear_SVM.fit(X_train, y_train)
sigmoid_SVM.fit(X_train, y_train)
rbf_SVM.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_linear = linear_SVM.predict(X_test)
y_pred_sigmoid = sigmoid_SVM.predict(X_test)
y_pred_rbf = rbf_SVM.predict(X_test)

#Calculate the accuracy of our model
print("Accuracy (linear):",metrics.accuracy_score(y_test, y_pred_linear))
print("Accuracy (sigmoid):",metrics.accuracy_score(y_test, y_pred_sigmoid))
print("Accuracy (rbf):",metrics.accuracy_score(y_test, y_pred_rbf))