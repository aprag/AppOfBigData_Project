#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# The goal of this notebook is see how to clean the data and to do the feature engineering

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


test_df = pd.read_csv('../data/raw/application_test.csv')
train_df = pd.read_csv('../data/raw/application_train.csv')


# In[3]:


print(train_df.shape, test_df.shape)


# ## Delete missing values 
There is to much data with missing values so we start to drop the columns with more than 60% of missing values.
# In[4]:


# checking missing data in train_df

number = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum() / train_df.isnull().count() * 100).sort_values(ascending = False)

missing_train_df = pd.concat([number , percent] , axis = 1 , keys = ['Total' , 'Percent'])
print(missing_train_df.shape)
print(missing_train_df.head(40))


# In[5]:


# checking missing data in test_df
number = test_df.isnull().sum().sort_values(ascending = False)
percent = (test_df.isnull().sum() / test_df.isnull().count() * 100).sort_values(ascending = False)

missing_test_df = pd.concat([number , percent] , axis = 1 , keys = ['Total' , 'Percent'])
print(missing_test_df.shape)
print(missing_test_df.head(40))


# In[6]:


train_df['TARGET'].unique


# In[7]:


#we drop columns with more than 60% missing values
def dropna(df):
    mv=df.isna().sum()/df.shape[0]
    val=mv[mv>0.60]
    l=[i for i in val.index]
    dat=df.drop(l,axis=1)
    return dat


# In[8]:


train_df=dropna(train_df)


# In[9]:


test_df=dropna(test_df)


# In[10]:


print(train_df.shape, test_df.shape)


# In[21]:


for col in test_df.select_dtypes(include=[object]).columns:
    train_df[col] = train_df[col].fillna(train_df[col].mode(dropna=True)[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode(dropna=True)[0])


# In[12]:


for col in test_df.select_dtypes(include=[int,float]).columns:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())


# In[18]:


train_df.isnull().sum().sum()


# In[19]:


test_df.isnull().sum().sum()


# In[13]:


print(train_df.shape, test_df.shape)


# In[14]:


# Use dummies
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
target = train_df['TARGET']

train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)
train_df['TARGET'] = target


# In[15]:


print(train_df.shape, test_df.shape)


# In[16]:


train_df.to_csv(r'../data/processed/train.csv')


# In[17]:


test_df.to_csv(r'../data/processed/test.csv')

