# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:35:58 2020

@author: lyn
"""
import numpy as np
import pandas as pd

data=pd.read_csv(r'boston.csv')

print(data.head())

class LinearRegression:
    '''使用python语言实现线性回归算法。（梯度下降）'''
    def __init__(self,alpha,times):
        '''初始化方法。
        Parameters
        -----
        alpha:float
            学习率，用来控制步长，（权重调整的幅度）
        time:int
            循环迭代的次数。
        '''
        self.alpha=alpha
        self.times=times
    def fit(self,X,y):
        '''根据提供的训练数据，对模型进行训练
        Parameters
        -----
        X:类数组类型，形状【样本数量，特征数量】
            待训练的样本特征属性。（特征矩阵）
        y:类数组类型，形状【样本数量】
            目标值（标签信息）
'''
        X=np.asarray(X)
        y=np.asarray(y)
        #创建权重的向量，初始值为0,（或任何其他的值），长度比特征数量多1，（多的一个值就是截距）
        self.w_=np.zeros(1 + X.shape[1])
        #创建损失列表，用来保存每次迭代后的损失值，损失值计算：【预测值-真实值】得平方和 除以2
        self.lose_=[]
        
        #进行循环，每次迭代，在每次迭代的过程中，不断调整权重值，是的损失值不断减少
        for i in range(self.times):
            #计算预测值
            y_hat = np.dot(X,self.w_[1:])+self.w_[0]
            
            #计算真实值与预测值之间的差距。
            error = y - y_hat
            
            #将损失值加入到损失列表当中。
            self.lose_.append(np.sum(error**2)/2)
            
            #根据差距调整w_,根据公式： 调整为 权重（j） = 权重（j）+学习率 * sum((y-y_hat)*x(j))
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T,error)
            
    def predict(self,X):
        '''根据参数传递的样本，对样本数据进行预测
        Parameters
        -----
        X:类数组类型，形状【样本数量，特征数量】
            待测试的样本
        Returns
        -----
        result:数组类型
                预测的结果
        '''
        X=np.asarray(X)
        
        result=np.dot(X,self.w_[1:]+self.w_[0])
        
        return result
    
lr=LinearRegression(alpha=0.0005, times=20)
t= data.sample(len(data),random_state=0)

train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]

test_X = t.iloc[400:,:-1]
test_y= t.iloc[400:,-1]

lr.fit(train_X,train_y)
result = lr.predict(test_X)

print(np.mean((result-test_y)**2))

print('权重：=',lr.w_)
print('111',lr.lose_)



class StanderScaler:
    '''该类对数据进行标准化处理。'''
    def fit(self,X):
        '''根据传递的样本，计算每个特征列的均值与标准差
        
        Parameters
        -----
        X:类数组类型
            训练数据，用来计算均值与标准差
        '''
        X=np.asarray(X)
        #axis=0 按照列
        self.std_=np.std(X,axis=0)
        self.mean_=np.mean(X,axis=0)
        
    def transform(self,X):
        '''将给定的数据X，进行标准化处理，（将X的每一列都变成标准正太分布的数据）
        Parameters
        -----
        X:类数组类型
            带转换的数组
        Returns
        -----
        result:类数组类型
            参数转换成正太分布的结果
        '''
        return (X-self.mean_)/self.std_
    def fit_transform(self,X):
        '''对数据进行训练并转换，返回转换之后的结果
            Paramters
            -----
             X:类数组类型
            带转换的数组
        Returns
        -----
        result:类数组类型
            参数转换成正太分布的结果
        '''
        self.fit(X)
        return self.transform(X)
#为了避免每个特征数量级的不同，从而在梯度下降的过程中带来影响
#我们现在在考虑对每个特征进行标准化处理
lr=LinearRegression(alpha=0.0005, times=20)
t= data.sample(len(data),random_state=0)

train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]

test_X = t.iloc[400:,:-1]
test_y= t.iloc[400:,-1]

#对数据进行标准化处理
s=StanderScaler()
train_X=s.fit_transform(train_X)
test_X = s.transform(test_X)

s2=StanderScaler()
train_y=s2.fit_transform(train_y)
test_y=s2.transform(test_y)

lr.fit(train_X,train_y)
result = lr.predict(test_X)

print(np.mean((result-test_y)**2))

print('权重2：=',lr.w_)
print('222',lr.lose_)     



#可视化展示

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"]='SimHei'
mpl.rcParams["axes.unicode_minus"]=False
'''
plt.figure(figsize=(10,10))
#绘制预测值
plt.plot(result,"ro-",label="预测值")
#绘制真实值
plt.plot(test_y.values,"go--",label="真实值")
plt.title("线性回归预测-梯度下降")
plt.xlabel('样本序号')
plt.ylabel('房价')
plt.legend()
plt.show()

#绘制累计误差值
plt.plot(range(1, lr.times+1),lr.lose_,"o-")
'''
#因为房价分析设计多个维度，不方便进行可视化展示，为了实现可视化
#我们只选取其中的一个维度（RM），并会出直线，实现拟合
lr = LinearRegression(alpha=0.005, times=50)
t = data.sample(len(data),random_state=0)


train_X = t.iloc[:400,5:6]
train_y = t.iloc[:400,-1]

test_X = t.iloc[400:,5:6]
test_y= t.iloc[400:,-1]

#对数据进行标准化处理
s=StanderScaler()
train_X=s.fit_transform(train_X)
test_X = s.transform(test_X)

s2=StanderScaler()
train_y=s2.fit_transform(train_y)
test_y=s2.transform(test_y)

lr.fit(train_X,train_y)
result = lr.predict(test_X)


#直线拟合

plt.scatter(train_X['RM'], train_y)
print(lr.w_)
#构建方程  
x=np.arange(-5,5,0.1)
'''
y=-2.35400588e-14+  6.66133815e-16 * x
plt.plot(x,y,'r')
'''
#也可以这样做
plt.plot(x,lr.predict(x.reshape(-1,1)),'r')