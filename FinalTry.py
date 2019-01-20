import numpy as np

from PreProcess import PreProcess
import xgboost as xgb

p = PreProcess()

params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',  # 多分类的问题
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
    'eval_metric': 'rmse'
}
xgb.Booster()

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])



i = 1
predictday = 20

Xtrain, Wtrain, Ytrain, Xdev, Wdev, Ydev, Xtest, Wtest = p.generate_data4bignet(i, max([0, 36 * i-1]), 36*i+2, predictday, 60, return_weight=True)
dtrain = xgb.DMatrix(Xtrain.reshape(Xtrain.shape[0], -1), Ytrain)
dval = xgb.DMatrix(Xdev.reshape(Xdev.shape[0], -1), Ydev)
Xtestd = xgb.DMatrix(Xtest.reshape(Xtest.shape[0], -1))

watchlist = [(dtrain, 'train'), (dval, 'val')]

# cvresult = xgb.cv(params, Xd,  nfold=5,
#                   metrics='rmse', early_stopping_rounds=50)

model = xgb.train(params, dtrain, num_boost_round=100000, evals=watchlist)



