#!/usr/bin/env python
# coding: utf-8

# # [作業目標]
# - 仿造範例的 One Hot Encoding, 將指定的資料進行編碼

# # [作業重點]
# - 將 sub_train 進行 One Hot Encoding 編碼 (In[4], Out[4])

# In[1]:


import os
import numpy as np
import pandas as pd


# In[5]:


# 設定 data_path, 並讀取 app_train
f_app_train = os.path.join('application_train.csv')
app_train = pd.read_csv(f_app_train)


# ## 作業
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化

# In[14]:


sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


# In[15]:


"""
Your Code Here
"""
sub_train = pd.get_dummies(sub_train)


# In[16]:


sub_train.shape


# In[17]:


sub_train.head()


# In[ ]:




