#!/usr/bin/env python
# coding: utf-8

# # 處理 outliers
# * 新增欄位註記
# * outliers 或 NA 填補
#     1. 平均數 (mean)
#     2. 中位數 (median, or Q50)
#     3. 最大/最小值 (max/min, Q100, Q0)
#     4. 分位數 (quantile)

# # [作業目標]
# - 仿造範例的資料操作, 試著進行指定的離群值處理

# # [作業重點]
# - 計算 AMT_ANNUITY 的分位點 (q0 - q100) (Hint : np.percentile, In[3])
# - 將 AMT_ANNUITY 的 NaN 用中位數取代 (Hint : q50, In[4])
# - 將 AMT_ANNUITY 數值轉換到 -1 ~ 1 之間 (Hint : 參考範例, In[5])
# - 將 AMT_GOOD_PRICE 的 NaN 用眾數取代 (In[6])

# In[1]:


# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
# dir_data = './data/'


# In[2]:


f_app = os.path.join('application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# ## 1. 列出 AMT_ANNUITY 的 q0 - q100
# ## 2.1 將 AMT_ANNUITY 中的 NAs 暫時以中位數填補
# ## 2.2 將 AMT_ANNUITY 的數值標準化至 -1 ~ 1 間
# ## 3. 將 AMT_GOOD_PRICE 的 NAs 以眾數填補
# 

# In[4]:


"""
YOUR CODE HERE
"""
# 1: 計算 AMT_ANNUITY 的 q0 - q100
q_all = [i for i in range(1,101)]

q_list= [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in q_all]
q_list


# In[5]:


# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))

"""
Your Code Here
"""
q_50 = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 50)
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50

print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))


# ### Hints: Normalize function (to -1 ~ 1)
# $ y = 2*(\frac{x - min(x)}{max(x) - min(x)} - 0.5) $

# In[6]:


# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

def normalize_value(x):
    """
    Your Code Here, compelete this function
    """
    maximum = max(x)
    minimum = min(x)
    y = ((x-minimum)/(maximum-minimum)-0.5) * 2
    
    return y

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
app_train['AMT_ANNUITY_NORMALIZED'].describe()


# In[16]:


# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
"""
Your Code Here
"""
from scipy.stats import mode
value_most = mode(app_train[~app_train['AMT_GOODS_PRICE'].isnull()]['AMT_GOODS_PRICE'])
print(value_most)

mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))


# In[ ]:




