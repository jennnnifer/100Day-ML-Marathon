#!/usr/bin/env python
# coding: utf-8

# # [作業目標]
# - 使用 Day 17 剛學到的方法, 對較完整的資料生成離散化特徵
# - 觀察上述離散化特徵, 對於目標值的預測有沒有幫助

# # [作業重點]
# - 仿照 Day 17 的語法, 將年齡資料 ('DAYS_BIRTH' 除以 365) 離散化
# - 繪製上述的 "離散化標籤" 與目標值 ('TARGET') 的長條圖

# In[1]:


# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
# dir_data = './data/'


# ### 之前做過的處理

# In[2]:


# 讀取資料檔
f_app_train = os.path.join('application_train.csv')
app_train = pd.read_csv(f_app_train)
app_train.shape


# In[3]:


# 將只有兩種值的類別型欄位, 做 Label Encoder, 計算相關係數時讓這些欄位可以被包含在內
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
app_train.head()


# In[4]:


# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


# ## 練習時間
# 參考 Day 17 範例程式，離散化你覺得有興趣的欄位，並嘗試找出有趣的訊息

# In[5]:


test = app_train.iloc[:,[0,1,7]]
test['AMT_INCOME_TOTAL'] = pd.qcut(app_train['AMT_INCOME_TOTAL'], q=10)
test['AMT_INCOME_TOTAL'].value_counts()


# In[6]:


test.head()


# In[ ]:




