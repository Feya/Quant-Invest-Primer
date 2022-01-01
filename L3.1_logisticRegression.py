#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 09:59:58 2021

This code performs the following tasks:
    1. Load in the bank customer data
    2. Run logistic regression
       background knowledge for logistic regression check any statistic book
    3. Check model performance using ROC:  
       background knowledge for ROC: 
        https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF 
    4. Dataset comes from the UCI Machine Learning repository. It is about 
       direct marketing campaigns (phone calls) of a Portuguese banking institution. 
       The classification goal is to predict whether the client will subscribe
       (1/0) to a term deposit (variable y).

@author: fanyang
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

#Inputdata = pd.DataFrame.from_csv('BankCustomerLogisticRExample.csv')
Inputdata = pd.read_excel('BankCustomerLogisticRExample.xlsx',
                          sheet_name='BankCustomerLogisticRExample')
Inputdata.head()

# pick X variables as those not named 'y'
varnames = Inputdata.columns.values.tolist()
Yvarname = ['y']
Xvarname = [i for i in varnames if i not in Yvarname]

X=Inputdata[Xvarname]
Y=Inputdata[Yvarname]

# Implement the model
import statsmodels.api as sm
# Logistic regression model fitting
logit_model = sm.Logit(Y,X)
result = logit_model.fit()
print(result.summary())

# Split X and Y into random train and test subsets and optionally subsampling
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train, Y_train)

# Predict the test set results and calculate the accuracy
Y_pred = logreg.predict(X_test)
print("Number of predicted hits is : ",sum(Y_pred),"out of ",len(Y_pred))
Accuracy = logreg.score(X_test,Y_test)
print("\nAccuracy of logistic regression classifier on test set : {:.2f}".format(Accuracy))

# background knowledge about confusion matrix and ROC curve: 
#    https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF    
# confusion matrix and manually calculate accuracy of the model
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print("\nConfusion matrix is : \n",confusion_matrix)
tn = confusion_matrix[0,0] #true negative
tp = confusion_matrix[1,1] #true positive
fn = confusion_matrix[1,0] #false negative
fp = confusion_matrix[0,1] #false positive
totaldatapoint=Y_test.shape[0]
accuracy_calc = (tn+tp)/totaldatapoint

# classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Calculate the ROC curve
prob = logreg.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(Y_test,prob)

# calculate the AUC of ROC from prediction scores
logit_roc_auc = roc_auc_score(Y_test, prob)
print('\n Area Under the Receiver Operating Characteristic Curve: {:.2f}'.format(logit_roc_auc))

# plot the ROC curve
plt.figure()
plt.plot(fpr,tpr,label='Logistic Regression (area=%0.2f' % logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



