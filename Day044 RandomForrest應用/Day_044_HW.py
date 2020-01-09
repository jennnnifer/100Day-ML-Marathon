#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 確保你了解隨機森林模型中每個超參數的意義，並觀察調整超參數對結果的影響

# ## 作業
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較

# In[28]:


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# In[29]:


data = datasets.load_boston()
X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)


# In[49]:


# 使用 20 顆樹，每棵樹的最大深度為 4
clf = RandomForestClassifier(n_estimators=20, max_depth=4)
clf.fit(X_train, y_train.astype('int'))
pred = clf.predict(X_test)
print("Acc: ", sum(y_test == pred)/len(y_test))


# In[52]:


# 使用 100 顆樹，每棵樹的最大深度為 6
clf = RandomForestClassifier(n_estimators=100, max_depth=6)
clf.fit(X_train, y_train.astype('int'))
pred = clf.predict(X_test)
print("Acc: ", sum(y_test == pred)/len(y_test))


# In[57]:


# 使用 1000 顆樹，每棵樹的最大深度為 10
clf = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf.fit(X_train, y_train.astype('int'))
pred = clf.predict(X_test)
print("Acc: ", sum(y_test == pred)/len(y_test))


# In[68]:


from sklearn.tree import DecisionTreeClassifier


# In[69]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train.astype(int))
pred = dt.predict(X_test)
sum(y_test==pred)/len(y_test)


# In[ ]:




