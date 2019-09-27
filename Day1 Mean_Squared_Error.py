#!/usr/bin/env python
# coding: utf-8

# # [作業1]

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


w = 3
b = 0.5

# np.linspace(0, 100, 101)是指 0~100 劃分成 101 個刻度(含頭尾), 所也就是 0, 1, 2,...,100 這 101 個數
# 這時候, x_lin 因為要記錄不只一個數, 因為 np.linspace() 傳回的是一個 Array, 所以 x_lin 就變成 Array 了
x_lin = np.linspace(0, 100, 101)

# np.random.randn() 就是 numpy.random.randn(), 會隨機傳回標準常態分布的取樣值
# np.random.randn(101) 表示取樣了101次, 型態是 Array, 所以其他 + 與 * 的部分都是 Array 的加與乘, 一行就計算了101筆資料
# 所以最後的結果 y, 也是一個長度 101 的 Array
y = (x_lin + np.random.randn(101) * 5) * w + b

# 這邊就是將 x_lin 以及剛剛算完的 y, 當作座標值, 將101個點在平面上畫出來
# b. : b 就是 blue, 點(.) 就是最小單位的形狀, 詳細可以查 matplotlib 的官方說明
plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()


# In[3]:


# 這邊的 y_hat, 就沒有隨機的部分了, 也就是下圖中的紅色實線部分
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
# 上面的 'b.' 是藍色點狀, 下面的 'r-' 是紅色線狀, label 是圖示上的名稱
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()


# In[4]:


def mean_absolute_error(y, yp):
#     計算 MAE
#     Args:
#         - y: 實際值
#         - yp: 預測值
#     Return:
#         - mae: MAE
    # MAE : 將兩個陣列相減後, 取絕對值(abs), 再將整個陣列加總成一個數字(sum), 最後除以y的長度(len), 因此稱為"平均絕對誤差"
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

# 呼叫上述函式, 傳回 y(藍點高度)與 y_hat(紅線高度) 的 MAE
MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))


# In[6]:


def mean_squared_error(y, yp):
    mse = sum((y - yp) ** 2)/len(y)
    return mse

MSE = mean_squared_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MSE))


# # [作業2]

# ## Data: Digit Recognizer 
# ### Learn computer vision fundamentals with the famous MNIST data
# 
# Q: 你選的這組資料為何重要<br/>
# A: 判斷手寫資料, 未來可運用在圖片辨識上, 例如: 車牌辨識<br/>
# <br/>
# Q: 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)<br/>
# A: 資料是由人手寫0~9的圖片<br/>
# <br/>
# Q: 蒐集而來的資料型態為何<br/>
# A: 下載下來的CSV檔可以用pandas讀入, 當中的每一筆資料型態為INT<br/>
# <br/>
# Q: 這組資料想解決的問題如何評估<br/>
# A: 這個題目是有正確答案的, 只需將output與label做對比即可知道正確率<br/>

# # [作業3]

# ### 想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：
# 
# 1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)<br/>
# 核心問題為提升業績, 對自由載客車隊而言, 業績即為載客數X載客營收<br/>
# <br/>
# 2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)<br/>
# 我們可以根據過去載客資料蒐集: 乘客上車地點, 上車時間, 乘車距離<br/>
# 統計出在何時何地載客需求變化<br/>
# <br/>
# 3. 蒐集而來的資料型態為何<br/>
# 蒐集而來的資料型態如下: <br/>
# 上車地點: STR<br/>
# 上車時間: INT(TIMESTAMP)<br/>
# 乘車距離: INT/FLOAT<br/>
# <br/>
# 4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)<br/>
# 根據分析後的結果, 將司機依照需求量分配到不同時間地點<br/>
# 觀察在相同時間內, 載客數與載客營收是否有提升<br/>

# In[ ]:





# In[ ]:




