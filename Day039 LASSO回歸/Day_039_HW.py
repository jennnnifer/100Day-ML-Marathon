#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 清楚了解 L1, L2 的意義與差異為何，並了解 LASSO 與 Ridge 之間的差異與使用情境

# ## 作業

# 請閱讀相關文獻，並回答下列問題
# 
# [脊回歸 (Ridge Regression)](https://blog.csdn.net/daunxx/article/details/51578787)
# [Linear, Ridge, Lasso Regression 本質區別](https://www.zhihu.com/question/38121173)
# 
# 1. LASSO 回歸可以被用來作為 Feature selection 的工具，請了解 LASSO 模型為什麼可用來作 Feature selection
# 2. 當自變數 (X) 存在高度共線性時，Ridge Regression 可以處理這樣的問題嗎?
# 
ANS: 
1. LASSO回歸的目標函數為LR + L1, 此時若經過幾次回歸後, 係數w降為0, 則可以捨棄該Feature, 因此可以做Feature selection
2. 