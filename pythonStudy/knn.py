# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
"""
Created on Sat Mar 21 14:45:51 2020

@author: MaiBenBen
"""
class KNN:
    '''使用python语言实现KNN近邻算法（实现分类）'''
    def __init__(self,k):
        '''初始化方法
        Parameters
        ----------
        k:int
            邻居的个数
        '''
        self.k=k
    def fit(self,X,y):
        '''训练的方法
        Parameters
        -----------
        X:类数组类型，形状为：[样本的数量，特征的数量]
            待训练的样本特征（属性）
            
        y:类数组类型，形状为:[样本数量]
            每个样本的目标值（标签）。
        
        '''
        #将X转换成ndarray数据类型
        self.X=np.asarray(X)
        self.y=np.asarray(y)
    
    def predict(self,X):
        '''根据参数传递的样本，对样本数据进行预测
        Parameters
        -----------
        X:类数组类型，形状为：[样本的数量，特征的数量]
            待训练的样本特征（属性）
        Return
        ----------
        result:数组类型
            预测的结果。
        '''
        X=np.asarray(X)
        result=[]
        # 对ndarray数据进行遍历，每次取数据中的一行
        for x in X:
            #axis 指定轴按照行求和。对于测试集中的每一个样本，依次与训练集中的所有样本求距离。
           dis=np.sqrt(np.sum((x - self.X)**2,axis=1))
           #返回数组排序后，每个元素在原数组（排序之前的数组）中的索引。
           index=dis.argsort()
           #进行截断，只取前k个元素。【取距离最近的k个元素的索引】
           index=index[:self.k]
           #返回数组中每个元素出现的次数，元素必须是非负的整数。
           count=np.bincount(self.y[index],weights=1/dis[index])
           #返回ndarray数组中，值最大的元素对应的索引，该索引就是我们判定的类别。
           #最大的元素索引，就是出现次数最多的元素。
           result.append(count.argmax())
        return np.asarray(result)
           
data=pd.read_csv('iris.csv', header=None)    
data[4]=data[4].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
#print(data)

#提取每个类别的鸢尾花数据
t0 = data[data[4]==0]
t1 = data[data[4]==1]
t2 = data[data[4]==2]

#对每个类别的数据进行洗牌random_state相同是随机的情况一样，每次都是这么随机
t0 = t0.sample(len(t0),random_state=0)
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)

#对数据进行切割 iloc 是切割前40行不包括四十，列数是最后一列之后的所有列  axis =0按照纵向的方式进行拼接
train_X = pd.concat([t0.iloc[:40,:-1],t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t0.iloc[:40,-1],t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t0.iloc[40:,:-1],t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t0.iloc[40:,-1],t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)
#创建KNN对象，进行训练与测试
knn=KNN(k=3)
#进行训练
knn.fit(train_X,train_y)
#进行测试，获得测试结果
result = knn.predict(test_X)
#print(result)
#print(test_y)

print(np.sum(result==test_y))
print(len(result))
print("正确率是：",(np.sum(result==test_y)/len(result))*100,"%")

"""
结果可视化展示
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

#默认情况下，matplotlib 不支持中文显示，我们需要设置一下

#设置字体为黑体，以支持中文显示
mpl.rcParams["font.family"]="SimHei"
#设置在中文字体时，能够正常的显示负号（-）
mpl.rcParams["axes.unicode_minus"]=False

#{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
#设置画布的大小
plt.figure(figsize=(10,10))
#绘制 训练集的数据
plt.scatter(x=t0[0][:40],y=t0[2][:40],color='y',label='Iris-setosa')
plt.scatter(x=t1[0][:40],y=t1[2][:40],color='g',label='Iris-versicolor')
plt.scatter(x=t2[0][:40],y=t2[2][:40],color='b',label='Iris-virginica')

#绘制测试集数据  分为正确数据和错误数据
right = test_X[result==test_y]
wrong = test_X[result!=test_y]
plt.scatter(x=right[0],y=right[2],color='c',marker='x',label="right")
plt.scatter(x=wrong[0],y=wrong[2],color='r',marker='>',label="wrong")
plt.xlabel('花萼的长度')
plt.ylabel('花瓣的长度')
plt.legend(loc="best")
plt.show()



        
        
        

