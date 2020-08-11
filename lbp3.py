import numpy as np
import random
import pickle
import platform
import os
#加载序列文件
def load_pickle(f):
    version=platform.python_version_tuple()#判断python的版本
    if version[0]== '2':
        return pickle.load(f)
    elif version[0]== '3':
        return pickle.load(f,encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))
#处理原数据,cifar-100中有两个标签：fine_labels:表示0-99个精确标签；coarse_labels表示0-19个粗标签
def load_CIFAR_train(filename):
    with open(filename,'rb') as f:
        datadict=load_pickle(f)
        X=datadict['data']
        Y=datadict['fine_labels']
        X=X.reshape(50000,3,32,32).transpose(0,2,3,1).astype("float")
        #reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y=np.array(Y)
        return X,Y
def load_CIFAR_test(filename):
    with open(filename,'rb') as f:
        datadict=load_pickle(f)
        X=datadict['data']
        Y=datadict['fine_labels']
        X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        #reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y=np.array(Y)
        return X,Y

#返回可以直接使用的数据集
def load_CIFAR100(ROOT):
    Xtr,Ytr = load_CIFAR_train(os.path.join(ROOT,'train'))
    Xte,Yte=load_CIFAR_test(os.path.join(ROOT,'test'))
    return Xtr,Ytr,Xte,Yte

datasets = 'D:/picture-data/cifar-100-python'
x_train,y_train,x_test,y_test = load_CIFAR100(datasets)
print('x_train shape:%s, y_train shape:%s' % (x_train.shape, y_train.shape))
print('x_test shape:%s, y_test shape:%s' % (x_test.shape, y_test.shape))