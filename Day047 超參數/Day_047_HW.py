#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 了解如何使用 Sklearn 中的 hyper-parameter search 找出最佳的超參數

# ### 作業
# 請使用不同的資料集，並使用 hyper-parameter search 的方式，看能不能找出最佳的超參數組合

# In[8]:


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd


# In[2]:


data = datasets.load_boston()
X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=4)


# In[13]:


clf = GradientBoostingClassifier()

learning_rate = [0.05, 0.1, 0.2, 0.3]
n_estimators = [100, 300, 500]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

grid_search = GridSearchCV(clf, param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_result = grid_search.fit(X_train, y_train.astype(int))


# In[14]:


print(grid_result.best_score_, grid_result.best_params_)


# In[ ]:





# In[ ]:





# In[ ]:




