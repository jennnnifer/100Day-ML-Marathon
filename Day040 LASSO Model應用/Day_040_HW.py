#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 使用 Sklearn 中的 Lasso, Ridge 模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，也請了解模型中各項參數的意義。
# 
# 機器學習的模型非常多種，但要訓練的資料多半有固定的格式，確保你了解訓練資料的格式為何，這樣在應用新模型時，就能夠最快的上手開始訓練！

# ## 練習時間
# 試著使用 sklearn datasets 的其他資料集 (boston, ...)，來訓練自己的線性迴歸模型，並加上適當的正則化來觀察訓練情形。

# In[14]:


from sklearn import datasets, model_selection, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[8]:


boston = datasets.load_boston()

X = pd.DataFrame(boston.data)
y = pd.DataFrame(boston.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[15]:


# LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[23]:


lr.coef_


# In[22]:


# LASSO
la = Lasso(0.1)
la.fit(X_train, y_train)
y_pred = la.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[26]:


# Ridge
ri = Ridge(0.1)
ri.fit(X_train, y_train)
y_pred = ri.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[27]:


ri.coef_


# In[ ]:




