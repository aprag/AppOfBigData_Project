#!/usr/bin/env python
# coding: utf-8

# # Models Training

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


# ### Xgboost Classifier

# In[4]:


from xgboost import XGBClassifier


# In[19]:


XGB = XGBClassifier(objective='binary:logistic', eval_metric="logloss", use_label_encoder=False)
XGB.fit(X_train, y_train)


# In[20]:


XGBpred = XGB.predict(X_test)

print(XGB.score(X_test,y_test))


# In[21]:


from sklearn.metrics import confusion_matrix

XGBmatrix = confusion_matrix(y_test,XGBpred)


# In[22]:


sb.heatmap(XGBmatrix,annot=True, fmt='g', cmap='Blues')


# In[23]:


print("Accuracy : " + str(accuracy_score(y_test, XGBpred)))


# In[24]:


print("Precision : " + str(precision_score(y_test, XGBpred)))


# ### RandomForest Classifier

# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)


# In[12]:


RFCpred = RFC.predict(X_test)
print(RFC.score(X_test,y_test))


# In[13]:


RFCmatrix = confusion_matrix(y_test, RFCpred)
sb.heatmap(RFCmatrix,annot=True, fmt ='g',  cmap='Blues')


# In[14]:


print("Accuracy : " + str(accuracy_score(y_test, RFCpred)))


# In[15]:


print("Precision : " + str(precision_score(y_test, RFCpred)))


# ### GradientBossting Classifier

# In[25]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  confusion_matrix

GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)


# In[26]:


GBCpred = GBC.predict(X_test)
print(GBC.score(X_test,y_test))


# In[27]:


from sklearn.metrics import confusion_matrix

cmGBC = confusion_matrix(y_test,GBCpred)

sb.heatmap(cmGBC,annot=True, fmt='g', cmap='Blues')


# In[28]:


print("Accuracy : " + str(accuracy_score(y_test, GBCpred)))


# In[29]:


print("Precision : " + str(precision_score(y_test, GBCpred)))

