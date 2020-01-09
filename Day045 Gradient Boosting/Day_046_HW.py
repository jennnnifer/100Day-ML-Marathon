#!/usr/bin/env python
# coding: utf-8

# ### 作業
# 目前已經學過許多的模型，相信大家對整體流程應該比較掌握了，這次作業請改用**手寫辨識資料集**，步驟流程都是一樣的，請試著自己撰寫程式碼來完成所有步驟

# In[1]:


from sklearn import datasets, metrics
digits = datasets.load_digits()


# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
X = pd.DataFrame(digits.data)
y = pd.DataFrame(digits.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)


# In[15]:


from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Acc: ", sum(y_test[0].values==pred)/len(y_test))


# In[ ]:




