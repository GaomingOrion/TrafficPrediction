from keras.layers import*
from keras.models import*
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import pandas as pd

import matplotlib.pyplot as plt

from random import shuffle

dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228))))

# 选出每个station距离小于5km的其他station
cand_list = [[k] for k in range(228)]
for i in range(228):
    for j in range(i+1, 228):
        if dis_mat[i, j] < 5000:
            cand_list[i].append(j)
            cand_list[j].append(i)

EPOCHS = 50
BATCH_SIZE = 128
WINDOW = 3

def each_lstm_model(i, loadpath=None, plot=True):
    # 搭建网络
    InputDim = len(cand_list[i])*WINDOW
    print('特征个数为%i'%InputDim)
    model = Sequential()
    model.add(LSTM(20, input_shape=(InputDim, 12-WINDOW+1), dropout=0.2, return_sequences=True))
    model.add(LSTM(10, dropout=0.2))
    model.add(Dense(1))
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    if loadpath:
        model.load_weights(loadpath)
    print(model.summary())

    #训练模型
    checkpointer = ModelCheckpoint(filepath='SelectModel/checkpoint-%i-{epoch:02d}e-val_mse_{val_mean_squared_error:.5f}.hdf5'%i,
                                   monitor='val_mean_squared_error', save_best_only=True)
    Xtrain_station = flatten_bylag(Xtrain[:, cand_list[i], :], WINDOW)
    Xtest_station = flatten_bylag(Xtest[:, cand_list[i], :], WINDOW)
    history = model.fit(Xtrain_station, Ytrain[:, i], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
              validation_data=(Xtest_station, Ytest[:, i]), callbacks=[checkpointer])

    if plot:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.show()
    return model, history

def each_gru_model(i, loadpath=None, plot=True):
    # 搭建网络
    InputDim = len(cand_list[i])*WINDOW
    print('特征个数为%i'%InputDim)
    model = Sequential()
    model.add(GRU(30, input_shape=(InputDim, 12-WINDOW+1), dropout=0.2, return_sequences=True))
    model.add(GRU(10, dropout=0.2))
    model.add(Dense(1))
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    if loadpath:
        model.load_weights(loadpath)
    print(model.summary())

    #训练模型
    checkpointer = ModelCheckpoint(filepath='SelectModel/checkpoint-%i-{epoch:02d}e-val_mse_{val_mean_squared_error:.5f}.hdf5'%i,
                                   monitor='val_mean_squared_error', save_best_only=True)
    Xtrain_station = flatten_bylag(Xtrain[:, cand_list[i], :], WINDOW)
    Xtest_station = flatten_bylag(Xtest[:, cand_list[i], :], WINDOW)
    history = model.fit(Xtrain_station, Ytrain[:, i], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
              validation_data=(Xtest_station, Ytest[:, i]), callbacks=[checkpointer])

    if plot:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.show()
    return model, history


def flatten_bylag(Xdata, window=3):
    if window == 1:
        return Xdata
    data_shape = Xdata.shape
    res = [[X[:, i:(i+window)] for i in range(data_shape[2]-window+1)] for X in Xdata]
    res = np.array(res).reshape(data_shape[0], data_shape[1]*window, data_shape[2]-window+1)
    return res

if True:
    predictday = 14
    ## 用train数据集进行训练

    # test = list(range(34))
    # shuffle(test)
    # test = test[:2]
    test = [3, 20]
    # 读取训练数据
    Xtrain = []
    Ytrain = []
    for k in [i for i in range(34) if i not in test]:
        df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
        df = np.array(df).transpose()
        df = (df - 50) * 0.01
        Xtrain += [df[:, i:(i + 12)] for i in range(288 - predictday)]
        Ytrain += [df[:, i + predictday] for i in range(288 - predictday)]
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    print('训练数据shape', Xtrain.shape, Ytrain.shape)

    # 读取测试数据，用来估计mse
    Xtest = []
    Ytest = []
    for k in test:
        df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
        df = np.array(df).transpose()
        df = (df - 50) * 0.01
        Xtest += [df[:, i:(i + 12)] for i in range(288 - predictday)]
        Ytest += [df[:, i + predictday] for i in range(288 - predictday)]
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    print('测试数据shape', Xtest.shape, Ytest.shape)

    model = each_gru_model(117)
    #model = each_rnn_model(1, loadpath='SelectModel/checkpoint-1-31e-val_mse_0.00241.hdf5')
