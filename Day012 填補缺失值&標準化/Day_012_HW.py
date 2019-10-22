#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察填補缺值以及 標準化 / 最小最大化 對數值的影響

# # [作業重點]
# - 觀察替換不同補缺方式, 對於特徵的影響 (In[4]~In[6], Out[4]~Out[6])
# - 觀察替換不同特徵縮放方式, 對於特徵的影響 (In[7]~In[8], Out[7]~Out[8])

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# data_path = 'data/'
df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


# In[2]:


#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')


# In[3]:


# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0]
df.head()


# In[39]:


df.median()


# In[36]:


df.mean()


# In[29]:


df.fillna(list(mode(df)[0][0]))


# # 作業1
# * 試著在補空值區塊, 替換並執行兩種以上填補的缺值, 看看何者比較好?

# In[40]:


# 空值補 -1, 做羅吉斯迴歸

df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("填補-1 | R-Square: %f"%(r))

df_m1 = df.fillna(0)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("填補0 | R-Square: %f"%(r))

df_m1 = df.fillna(df.mean())
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("填補平均值 | R-Square: %f"%(r))

df_m1 = df.fillna(df.median())
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("填補中位數 | R-Square: %f"%(r))


# # 作業2
# * 使用不同的標準化方式 ( 原值 / 最小最大化 / 標準化 )，搭配羅吉斯迴歸模型，何者效果最好?

# In[43]:


"""
Your Code Here
"""
df_m1 = df.fillna(df.median())


train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("原值 | R-Square: %f"%(r))

def std(x):
    return (x-np.mean(x)) / np.std(x)
df_m1 = df.fillna(df.median())
df_m1 = std(df_m1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("標準化 | R-Square: %f"%(r))

def max_min(x):
    max_ = max(x)
    min_ = min(x)
    return (x - min_) / (max_ - min_)
df_m1 = df.fillna(df.median())
df_m1 = std(df_m1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
r = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print("最大最小化 | R-Square: %f"%(r))


# In[ ]:




