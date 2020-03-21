# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:00:33 2020

@author: lyn
"""
import numpy as np
import pandas as pd

data = pd.read_csv(r'iris.csv',header=None)
#print(data)
#删除不需要的ID和Species列（特征），因为现在进行回归预测，类别信息就没有用处了

data.drop([4],axis=1,inplace=True)
#删除重复记录
data.drop_duplicates(inplace=True)

class KNN:
    '''使用python实现近邻算法，（回归预测）
    该算法用于回归预测，根据前三个特征属性，寻找最近的k个邻居，然后在根据k个邻居的第四个特征
    属性，去预测当前样本的第四个特征值。
    '''
    def __init__(self,k):
        '''初始化方法
        Parameters
        ------
        k:int
            邻居的个数        
        '''
        self.k=k
    def fit(self,X,y):
        '''训练方法
        Parameters
        ------
        X:类数组类型(特征矩阵)，形状为【样本数量，特征数量】
            待训练的特征特征（属性）
        y:类数组类型（目标标签）。形状为【样本数量】
            每个样本的目标值（标签）
            '''
        #注意，将X与y转换成ndarray数组的形式，方便统一进行操作
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        
    def predict(self,X):
        '''根据参数传递的X，对样本数据进行预测
        Paramters:
        -----
        X:类数组类型。形状为【样本数量，特征数量】
            待测试的样本特征
        Return
        --------
        result:数组类型
                预测的结果值
        '''
        #转换成数组类型
        X=np.asarray(X)
        #保存预测的结果值
        result=[]
        for x in X:
            #计算距离，（计算与训练集中每个X的距离）
            dis = np.sqrt(np.sum((x - self.X)**2,axis=1))
            #返回数组排序后，每个元素都在源数组中（排序之前的数组）的索引
            index = dis.argsort()
            #取前K个距离最近的索引（在原数组中的索引）。
            index = index[:self.k]
            #取得均值加入到返回的结果列表当中
            result.append(np.mean(self.y[index]))
        return result
    def predict2(self,X):
        '''根据参数传递的X，对样本数据进行预测(考虑权重)
        权重的计算方式：使用每个节点（邻居）距离的倒数/所有节点距离倒数之和。
        Paramters:
        -----
        X:类数组类型。形状为【样本数量，特征数量】
            待测试的样本特征
        Return
        --------
        result:数组类型
                预测的结果值
        '''
        #转换成数组类型
        X=np.asarray(X)
        #保存预测的结果值
        result=[]
        for x in X:
            #计算距离，（计算与训练集中每个X的距离）
            dis = np.sqrt(np.sum((x - self.X)**2,axis=1))
            #返回数组排序后，每个元素都在源数组中（排序之前的数组）的索引
            index = dis.argsort()
            #取前K个距离最近的索引（在原数组中的索引）。
            index = index[:self.k]
            
            #计算权重
            #1.计算所有邻居节点的倒数之和 注意 最后加上一个很小的值就是避免除数（距离）为零的情况
            s=np.sum(1/(dis[index]+0.0001))
            #2.使用每个节点的倒数，除以倒数之和，得到权重
            weight=(1/(dis[index]+0.0001))/s
            #3.使用邻居节点的标签值，乘以对应的权重，然后求和，得到最终的预测结果
            result.append(np.sum(self.y[index]*weight))

        return result

t=data.sample(len(data),random_state=0)
train_X=t.iloc[:120,:-1]
train_y=t.iloc[:120,-1]

test_X=t.iloc[120:,:-1]
test_y=t.iloc[120:,-1]

knn = KNN(k=12)

knn.fit(train_X,train_y)
#未加权重的求解
#result=knn.predict(test_X)
#添加权重的求解
result=knn.predict2(test_X)
print(result)
np.mean((result-test_y)**2)

#将结果进行可视化展示
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family']='SimHei'
mpl.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(10,10))
#绘制预测值  使用红色 圆圈 实线的形式
plt.plot(result,"ro-",label="预测值")
#绘制真实值 绿色 实心，虚线显示形式
plt.plot(test_y.values,"go--",label="真实值")
plt.title("KNN连续值预测展示")
plt.xlabel("节点序号")
plt.ylabel("花瓣宽度")
plt.legend()
plt.show()
            
            