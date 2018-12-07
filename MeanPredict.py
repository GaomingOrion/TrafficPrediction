import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

mse_mat = []
for predictday in [14, 17, 20]:
    Xtrain = []
    Ytrain = []
    for k in [i for i in range(34)]:
        df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
        df = np.array(df).transpose()
        Xtrain += [df[:, i:(i + 12)] for i in range(288 - predictday)]
        Ytrain += [df[:, i + predictday] for i in range(288 - predictday)]
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    print('训练数据shape', Xtrain.shape, Ytrain.shape)

    Yhat = []
    for i in range(228):
        # 估计mse
        Yhat.append(np.mean(Xtrain[:, i, :], axis=1))
    Yhat = np.array(Yhat).transpose()
    mse = np.mean(np.square(Yhat - Ytrain))
    print('>>>拟合所有模型完成！%ith timepoint--validation_mse: %f' % (predictday + 1, mse))
    mse_station = np.mean(np.square(Yhat - Ytrain), axis=0)
    mse_mat.append(mse_station)
    print('每个站点的mse为:')
    print(mse_station)

mse_mat = pd.DataFrame(np.array(mse_mat).transpose(), columns=('3', '6', '9'))
mse_mat.index = list(range(228))
mse_mat = mse_mat.round(2)
mse_mat.to_csv('MeanPredictMse.csv', header=True, index=True)
