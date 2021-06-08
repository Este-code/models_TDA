## IMPORT PACKAGES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# create dummy data and classifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# key metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
# roc and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# f score 
from sklearn.metrics import f1_score


## CREATE DUMMY DATA 

data_X, class_label = make_classification(n_samples = 1000, n_features = 5, n_classes = 2) # generates a random n-class classification problem
trainX, testX, trainy, testy = train_test_split(data_X, class_label, test_size=0.3) # creates training and test sets


## CREATE DESCISION TREE

model = DecisionTreeClassifier() 
model.fit(trainX, trainy)


## CREATE NEW PREDICTIONS
# ​
predictions = model.predict_proba(testX)[:, 1] 


################# TASK 1 #################

## CALCULATE ACCURACY
accuracy_score_mod1 = accuracy_score(testy, predictions)
print('Accuracy score mod1: %f'%accuracy_score_mod1)
## CREATE CONFUSION MATRIX
confusion_matrix_mod1 = confusion_matrix(testy, predictions)

sns.heatmap(pd.DataFrame(confusion_matrix_mod1), annot=True)
plt.show()
################# TASK 2 #################

TN, FP, FN, TP = confusion_matrix_mod1.ravel()

## CALCUATE PRECISION, RECALL AND SPECIFICITY

precision = TP/(TP+FP)
print('Precision mod1: %f'%precision)
recall = TP/(TP+FP)
print('Recall mod1: %f'%recall)
specificity = TN/(TN+FP)
print('Specificity mod1: %f'%specificity)
################# TASK 3 #################

## CALCULATE AUC
auc = roc_auc_score(testy, predictions)
print('AUC mod1: %f' %auc)

## PLOT ROC CURVE

fpr, tpr = roc_curve(testy,predictions)[0:2]
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

################# TASK 4 #################
# ​
## CALCULATE F SCORE 
f_score_fromula = (2*precision*recall)/(precision+recall)
print('F-score mod1 using formula: %f'%f_score_fromula)
f_score = f1_score(testy,predictions)
print('F-score mod1 using function: %f'%f_score)
# ​
################# TASK 5 #################
# ​
print("\n")
data_X_mod2, class_label_mod2 = make_classification(n_samples = 1000, n_features = 5, n_classes = 2, flip_y = 0.1) 
trainX_mod2, testX_mod2, trainy_mod2, testy_mod2 = train_test_split(data_X_mod2, class_label_mod2, test_size=0.3)

model_mod2 = DecisionTreeClassifier() 
model_mod2.fit(trainX_mod2, trainy_mod2)

predictions_mod2 = model_mod2.predict_proba(testX_mod2)[:, 1] 

accuracy_score_mod2 = accuracy_score(testy_mod2, predictions_mod2)
print('Accuracy score mod2: %f'%accuracy_score_mod2)

confusion_matrix_mod2 = confusion_matrix(testy_mod2, predictions_mod2)
sns.heatmap(confusion_matrix_mod2, annot=True)
plt.show()

TN_mod2, FP_mod2, FN_mod2, TP_mod2 = confusion_matrix_mod2.ravel()

precision_mod2 = TP_mod2/(TP_mod2+FP_mod2)
print('Precision mod2: %f'%precision_mod2)
recall_mod2 = TP_mod2/(TP_mod2+FP_mod2)
print('Recall mod2: %f'%recall_mod2)
specificity_mod2 = TN_mod2/(TN_mod2+FP_mod2)
print('Specificity mod2: %f'%specificity_mod2)

auc_mod2 = roc_auc_score(testy_mod2, predictions_mod2)
print('AUC mod2: %f'%auc_mod2)

fpr_mod2, tpr_mod2 = roc_curve(testy_mod2, predictions_mod2)[0:2]
plt.plot(fpr_mod2,tpr_mod2)
plt.title("ROC curve mod2")
plt.xlabel("FPR mod2")
plt.ylabel("TPR mod2")
plt.show()

f_score_fromula_mod2 = (2*precision_mod2*recall_mod2)/(precision_mod2+recall_mod2)
print('F-score mod2 using fomrula: %f'%f_score_fromula_mod2)
f_score_mod2 = f1_score(testy_mod2,predictions_mod2)
print('F-score mod2 using funciton: %f'%f_score_mod2)