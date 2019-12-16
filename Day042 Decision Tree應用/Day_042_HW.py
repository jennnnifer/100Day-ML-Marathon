#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 目前你應該已經要很清楚資料集中，資料的型態是什麼樣子囉！包含特徵 (features) 與標籤 (labels)。因此要記得未來不管什麼專案，必須要把資料清理成相同的格式，才能送進模型訓練。
# 今天的作業開始踏入決策樹這個非常重要的模型，請務必確保你理解模型中每個超參數的意思，並試著調整看看，對最終預測結果的影響為何

# ## 作業
# 
# 1. 試著調整 DecisionTreeClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型的結果進行比較

# In[10]:


import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# In[16]:


boston = datasets.load_boston()
X = pd.DataFrame(boston.data)
y = pd.DataFrame(boston.target)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[25]:


DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)


# In[28]:


metrics.mean_squared_error(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




