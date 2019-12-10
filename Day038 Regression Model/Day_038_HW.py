#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 使用 Sklearn 中的線性迴歸模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，也請了解模型中各項參數的意義

# ## 作業
# 試著使用 sklearn datasets 的其他資料集 (wine, boston, ...)，來訓練自己的線性迴歸模型。

# ### HINT: 注意 label 的型態，確定資料集的目標是分類還是回歸，在使用正確的模型訓練！

# In[30]:


from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


# In[3]:


wine = datasets.load_wine()
boston = datasets.load_boston()
breast_cancer = datasets.load_breast_cancer()


# In[39]:


X = pd.DataFrame(wine.data[:])
y = pd.DataFrame(wine.target[:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

y_pred = np.where(y_pred>=0.5, 1, 0)

# precision = metrics.precision_score(y_test, y_pred)
# recall = metrics.recall_score(y_test, y_pred)
# f1_score = metrics.f1_score(y_test, y_pred)

# print("precision: ", precision)
# print("recall: ", recall)
# print("f1_score: ", f1_score)
accuracy = sum((y_pred==y_test).values)/len(y_test)
print("ACC: ", accuracy)


# In[ ]:




