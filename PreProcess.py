import numpy as np
import pandas as pd
class PreProcess():
    def __init__(self):
        self.name = ''

    def select_nearest(self, dist=5000, return_num=False):
        dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228))))
        # 选出每个station距离小于dist的其他station
        cand_list = [[k] for k in range(228)]
        for i in range(228):
            for j in range(i+1, 228):
                if 1 < dis_mat[i, j] < dist:
                    cand_list[i].append(j)
                    cand_list[j].append(i)
        if return_num:
            num = []
            for i in range(228):
                #buf = [int(dist//1000 - dis_mat[i, x]//1000) for x in cand_list[i]]
                buf = [5 if x==i else 1 for x in cand_list[i]]
                res = []
                for i in range(len(buf)):
                    res += [i]*buf[i]
                num.append(res)
            return cand_list, num
        else:
            return cand_list


    def select_k_nearest(self, k=50):
        dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228))))
        # 选出每个station距离小于5km的其他station
        cand_list = []
        for i in range(228):
            cand = [i]
            num = 1
            top = dis_mat[i].argsort()
            for j in top:
                if dis_mat[i, j] > 1:
                    cand.append(j)
                    num += 1
                if num == k:
                    break
            cand_list.append(cand)
        return cand_list

    def readdata_old(self, predictday):
        # 读取训练数据
        X = []
        Y = []
        for k in range(34):
            df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            X += [df[:, i:(i + 12)] for i in range(288 - predictday)]
            Y += [df[:, i + predictday] for i in range(288 - predictday)]
        X, Y = np.array(X), np.array(Y)
        # 随机选取一部分作为验证集
        shuffle_index = np.array(list(range(X.shape[0])))
        # np.random.seed(200)
        #np.random.shuffle(shuffle_index)
        shuffle_index = shuffle_index[::-1]
        n_train = int(X.shape[0] * 0.85)
        Xtrain, Ytrain = X[shuffle_index[:n_train]], Y[shuffle_index[:n_train]]
        print('训练数据shape', Xtrain.shape, Ytrain.shape)
        Xdev, Ydev = X[shuffle_index[n_train:]], Y[shuffle_index[n_train:]]
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

    def readdata(self, predictday):
        # 读取训练数据
        Xtrain = []
        Ytrain = []
        test = np.array(range(34))
        #np.random.seed(1215)
        np.random.shuffle(test)
        test = test[:4]
        for k in [x for x in range(34) if x not in test]:
            df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            Xtrain += [df[:, i:(i + 12)] for i in range(288 - predictday)]
            Ytrain += [df[:, i + predictday] for i in range(288 - predictday)]
        Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)
        print('训练数据shape', Xtrain.shape, Ytrain.shape)

        # 读取验证数据
        Xdev = []
        Ydev = []
        for k in test:
            df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
            df = np.array(df).transpose()
            df = self.data_tf(df)
            Xdev += [df[:, i:(i + 12)] for i in range(288 - predictday)]
            Ydev += [df[:, i + predictday] for i in range(288 - predictday)]
        Xdev, Ydev = np.array(Xdev), np.array(Ydev)
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

    def generate_data4bignet(self, predictday, nearestnum=50):
        candlist = self.select_k_nearest(nearestnum)
        Xtrain, Ytrain, Xdev, Ydev, Xtest = self.readdata(predictday)
        Xtrain, Ytrain = self.transformXY4bignet(Xtrain, Ytrain, candlist, nearestnum)
        Xdev, Ydev = self.transformXY4bignet(Xdev, Ydev, candlist, nearestnum)
        Xtest = self.transformXY4bignet(Xtest)
        return Xtrain, Ytrain, Xdev, Ydev, Xtest

    def transformXY4bignet(self, X, Y, candlist, nearestnum):
        Xtrans, Ytrans = [], []
        for i in range(228):
            a = candlist[i][:nearestnum]
            Xtrans.append(X[:, a, :])
            Ytrans.append(Y[:, i])
        Xtrans = np.array(Xtrans).reshape(-1, nearestnum, 12)
        Ytrans = np.array(Ytrans).reshape(-1)
        return Xtrans, Ytrans

    def transformXtest4bignet(self, X, candlist, nearestnum):
        res = []
        for i in range(228):
            a = candlist[i][:nearestnum]
            res.append(X[:, a, :])
        res = np.array(res)
        return res

    def data_tf(self, df):
        return df*0.01

    def data_inv_tf(self, df):
        return 100*df

if __name__ == '__main__':
    p = PreProcess()
    Xtrain, Ytrain, Xdev, Ydev, Xtest = p.generate_data4bignet(14, 50)