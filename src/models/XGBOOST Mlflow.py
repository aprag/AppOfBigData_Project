#!/usr/bin/env python
# coding: utf-8

# # MLFlow 

# ## XGBOOST

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
import numpy as np
import pickle
import matplotlib.pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import mlflow
import mlflow.xgboost


# In[2]:


df_train = pd.read_csv('../data/processed/train.csv')
df_test = pd.read_csv('../data/processed/test.csv')


# In[3]:


X = df_train.drop(columns = ['TARGET'])
Y = df_train['TARGET']

df_test2 = df_test
X_train, X_test, y_train, y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[9]:


import xgboost as xgb
from xgboost import XGBClassifier
from mlflow.utils.environment import _mlflow_conda_env
import os
import warnings
import sys
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
def train_xgboost(lr, n_estim):
    
    with mlflow.start_run(run_name='xgboost'):
               
        xgb =XGBClassifier(learning_rate= lr, n_estimators= n_estim, seed= 42, subsample= 1, colsample_bytree= 1,max_depth= 3,scale_pos_weight=11)
        xgb.fit(X_train, y_train)
        mlflow.xgboost.autolog()
        
        y_pred_auc = xgb.predict_proba(X_test)[:,1]
        y_pred = xgb.predict(X_test)
      
        roc = roc_auc_score(y_test, y_pred_auc)*100
        acc = accuracy_score(y_test,y_pred)
       
        mlflow.log_metric("auc_roc",roc)
        mlflow.log_metric("accuracy_score",acc)
              
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('n_estimators', n_estim)
        mlflow.log_param('seed', 0)
        mlflow.log_param('subsample', 1)
        mlflow.log_param('colsamples_bytree', 1)
        mlflow.log_param('objective','binary:logistic')
        mlflow.log_param('max_depth', 3)
        mlflow.log_param('scale_pos_weight', 11)
        
        #log model
        mlflow.xgboost.log_model(xgb, "model")
        print("roc_auc",roc)
        print("accuracy_score",acc)
        


# In[10]:


train_xgboost(0.0001,100)

