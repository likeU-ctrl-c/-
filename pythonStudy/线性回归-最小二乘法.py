# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:00:43 2020

@author: lyn
"""
import numpy as np
import pandas as pd
'''
波士顿房价数据集字段说明
CRIM 房屋所在镇的犯罪率
ZN 面积大于25000平方英尺住宅所占的比例
INDUS 房屋所在镇非零售区域所占比例
CHAS 房屋是否位于河边，如果在河边值为1，否则值为0
NOX 一氧化氮的浓度
RM 平均房屋数量
AGE 1940年前建成房屋所占的比例
DIS 房屋距离波斯顿五大就业中心的加权距离
RAD 距离房屋最近的公寓
TAX 财产税额度
PTRATIO 房屋所在镇师生比例
B 计算公式：1000 * （房屋所在镇非没记人口所在比例 -0.63）**2
LSTAT  弱势群体人口所在比例
MEDV  房屋的平均价格
'''

data = pd.read_csv(r'boston.csv')

class LinearRegression:
    '''使用python实现的线性回归。（最小二乘法）'''
    
    def fit(self,X,y):
        '''根据提供的训练数据X，对模型进行训练。
        Parameters
        -----
        X：类数组类型，形状【样本数量，特征数量】
          特征矩阵，用来对模型进行训练。
        y:类数组类型，形状【样本数量】
        '''
        #X必须是一个完整的数组类型，不能是数组的一部分
        #说明：X是数组的一部分，而不是完整的对象数据（例如，X是由其他对象通过切片传递过来的
        #则无法完成矩阵的转换。
        #这里创建一个X的拷贝对象，避免转换矩阵的时候失败
        #转换成矩阵的形式
        X=np.asmatrix(X.copy())
        #y是个一维结构（行向量，或者列向量），一维结构可以不用进行拷贝
        #注意：我们现在要进行矩阵的运算，因此需要是二维的结构，我们通过reshape方法进行转换。一列的矩阵
        y=np.asmatrix(y).reshape(-1,1)
        #通过最小二乘公式，求解出最佳的权重值 X的转置 .I 是求幂
        self.w_=(X.T*X).I * X.T * y
    def predict(self,X):
        '''根据参数传递的样本X，对样本数据进行预测。
        Parameters
        -----
        X：类数组类型，形状【样本数量，特征数量】
          待遇测的样本特征（属性）。
        Return
        -----
        result:数组类型
                预测的结果
        '''
        #将X转换成矩阵，注意，需要对X进行拷贝
        X=np.asmatrix(X.copy()) 
        result=X*self.w_
        #将矩阵转换成ndarray数组，进行扁平化处理，然后返回结果
        #使用ravel可以将数组进行扁平化处理。
        return np.array(result).ravel()
'''
#不考虑截距的情况
t=data.sample(len(data),random_state=0)
train_X=t.iloc[:400,:-1]
train_y=t.iloc[:400,-1]

test_X=t.iloc[400:,:-1]
test_y=t.iloc[400:,-1]

lr=LinearRegression()
lr.fit(train_X,train_y)
result=lr.predict(test_X)

print(np.mean((result-test_y)**2))
#查看模型的权重值
print("模型的权重",lr.w_)
'''
#考虑截距的情况 增加一列，该列的所有值都是1
t=data.sample(len(data),random_state=0)
#可以增加一列
# t["Intercept"] =1
#按照习惯，截距作为w0,我们为之配上一个w0列数在最前面
new_columns=t.columns.insert(0,'Intercept')
#重新安排列的顺序，如果值为空，则使用fill_value 参数指定的值进行填充
t = t.reindex(columns=new_columns,fill_value=1)

train_X=t.iloc[:400,:-1]
train_y=t.iloc[:400,-1]

test_X=t.iloc[400:,:-1]
test_y=t.iloc[400:,-1]

lr=LinearRegression()
lr.fit(train_X,train_y)
result=lr.predict(test_X)

#结果可视化
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family']='SimHei'
mpl.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(10,10))
#绘制预测值  使用红色 圆圈 实线的形式
plt.plot(result,"ro-",label="预测值")
#绘制真实值 绿色 实心，虚线显示形式
plt.plot(test_y.values,"go--",label="真实值")
plt.title("线性回归预测-最小二乘法")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()
            
