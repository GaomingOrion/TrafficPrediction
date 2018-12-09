import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from PreProcess import PreProcess


def each_linear_model(Xtrain, Ytrain):
    model = linear_model.LinearRegression()
    model.fit(Xtrain, Ytrain)
    return model

def each_ridge_model(Xtrain, Ytrain):
    model = linear_model.RidgeCV(alphas=[0.1, 0.25, 0.5, 0.75, 1, 5, 10],
                                 fit_intercept=True, normalize=False)
    model.fit(Xtrain, Ytrain)
    return model

def each_lasso_model(Xtrain, Ytrain):
    model = linear_model.LassoCV(alphas=[0.00001, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001],
                                 max_iter=10000, fit_intercept=False, normalize=False)
    model.fit(Xtrain, Ytrain)
    print(model.alpha_)
    return model

def each_rf_model(Xtrain, Ytrain):
    model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=100, max_features=12)
    model.fit(Xtrain, Ytrain)
    return model


def each_xgb_model(Xtrain, Ytrain):
    param = {}
    dtrain = xgb.DMatrix(Xtrain)
    dtest = xgb.DMatrix(Xtest)
    model = None
    return model


if __name__ == '__main__':
    prepro = PreProcess()
    cand_list = prepro.select_nearest()
    Ysubmit = []
    mse_mat = []
    for predictday in [14, 17, 20]:
        Xtrain, Ytrain, Xdev, Ydev, Xtest = prepro.readdata(predictday)
        mse_station = []
        Vy = []
        # 拟合模型
        Yhat = []
        Vy = []
        for i in range(228):
            Xtrain0, Ytrain0 = Xtrain[:, cand_list[i], :].reshape(Xtrain.shape[0], -1), Ytrain[:, i]
            Xdev0, Ydev0 = Xdev[:, cand_list[i], :].reshape(Xdev.shape[0], -1), Ydev[:, i]
            Xtest0 = Xtest[:, cand_list[i], :].reshape(Xtest.shape[0], -1)

            model = each_linear_model(Xtrain0, Ytrain0)
            Yhat.append(model.predict(Xdev0))
            Vy.append(model.predict(Xtest0))

            if (i+1)%50 == 0:
                print('>>拟合完成%i个模型'%(i+1))
        Yhat = np.array(Yhat).transpose()
        mse = np.mean(np.square(prepro.data_inv_tf(Yhat) - prepro.data_inv_tf(Ydev)))
        print('>>>拟合所有模型完成！%ith timepoint--validation_mse: %f'%(predictday+1, mse))
        mse_station = np.mean(np.square(prepro.data_inv_tf(Yhat) - prepro.data_inv_tf(Ydev)), axis=0)
        mse_mat.append(mse_station)
        print('每个站点的mse为:')
        print(mse_station)
        Ysubmit.append(np.array(Vy))
    mse_mat = np.array(mse_mat)
    print('>>>拟合所有模型完成！validation_mse: %f'%np.mean(mse_mat))
    # mse_mat = pd.DataFrame(np.array(mse_mat).transpose(), columns=('3', '6', '9'))
    # mse_mat.index = list(range(228))
    # mse_mat = mse_mat.round(2)
    # mse_mat.to_csv('EachStationMse.csv', header=True, index=True)

    # 生成提交的csv文件
    # Ysubmit = data_inv_tf(np.array(Ysubmit))
    # res = pd.DataFrame(columns=['Id', 'Expected'])
    # timepoint = [15, 30, 45]
    # for d in range(80):
    #     for t in range(3):
    #         for i in range(228):
    #             res = res.append({'Id': '%i_%i_%i' % (d, timepoint[t], i), 'Expected': Ysubmit[t, i, d]},
    #                              ignore_index=True)
    # res.to_csv('prediction.csv', header=True, index=False)
