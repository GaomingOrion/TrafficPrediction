import numpy as np
import pandas as pd
class PreProcess():
    def __init__(self):
        self.name = ''

    def select_neareat(self):
        dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228))))
        # 选出每个station距离小于5km的其他station
        cand_list = [[k] for k in range(228)]
        for i in range(228):
            for j in range(i+1, 228):
                if dis_mat[i, j] < 5000:
                    cand_list[i].append(j)
                    cand_list[j].append(i)
        return cand_list

    def readdata(self, predictday):
        # 选取大部分训练集作为训练集
        test = np.array(range(34))
        np.random.shuffle(test)
        test = test[:4]
        # 读取训练数据
        Xtrain = []
        Ytrain = []
        for k in [i for i in range(34) if i not in test]:
            df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            Xtrain += [df[:, i:(i + 12)] for i in range(288 - predictday)]
            Ytrain += [df[:, i + predictday] for i in range(288 - predictday)]
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)
        print('训练数据shape', Xtrain.shape, Ytrain.shape)

        # 选取一部分训练集作为验证集，用来估计mse
        Xdev = []
        Ydev = []
        for k in test:
            df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            Xdev += [df[:, i:(i + 12)] for i in range(288 - predictday)]
            Ydev += [df[:, i + predictday] for i in range(288 - predictday)]
        Xdev = np.array(Xdev)
        Ydev = np.array(Ydev)
        print('验证数据shape', Xdev.shape, Ydev.shape)

        # 读取test数据集，用来之后提交预测
        Xtest = []
        for k in range(80):
            df = pd.read_csv('data/test/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            Xtest.append(df)
        Xtest = np.array(Xtest)
        return Xtrain, Ytrain, Xdev, Ydev, Xtest

    def data_tf(self, df):
        return df*0.01

    def data_inv_tf(self, df):
        return 100*df
