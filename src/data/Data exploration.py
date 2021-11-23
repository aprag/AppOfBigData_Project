#!/usr/bin/env python
# coding: utf-8

# # DATA EXPLORATION 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()


test_df = pd.read_csv('../data/raw/application_test.csv')
train_df = pd.read_csv('../data/raw/application_train.csv')


# In[2]:


print("test shape :")
test_df.shape


# In[3]:


print("train shape :")
train_df.shape


# In[4]:


test_df.head()


# In[5]:


train_df.head()


# In[6]:


train_df.columns


# In[7]:


test_df.columns


# ## Missing values :

# In[8]:


train_df.isnull()


# In[9]:


train_df.isnull().sum().sum()


# In[10]:


test_df.isnull().sum().sum()


# In[11]:


plt.ylabel('Instances')
plt.xlabel('TARGET value')
plt.title('Target Variable Distribution (Training Dataset)')
sns.countplot(x='TARGET', data=train_df);

TARGET' value at 0 constitue approximately 90% the Training dataset
# In[12]:


correlations = train_df.corr()['TARGET'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(10))
print('\nMost Negative Correlations: \n', correlations.head(10))


# In[ ]:




