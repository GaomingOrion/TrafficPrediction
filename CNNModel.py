from keras.layers import*
from keras.models import*
from keras.callbacks import ModelCheckpoint
from keras import optimizers
def each_cnn_model(i, loadpath=None, plot=False):
    # 搭建网络
    InputDim = len(cand_list[i])
    print('特征个数为%i'%InputDim)
    model = Sequential()
    # model.add(Conv1D(filters=10, kernel_size=3, input_shape=(12, InputDim),
    #                  kernel_regularizer=regularizers.l1(0.01)))
    # model.add(Dropout(0.2))
    # #model.add(BatchNormalization())
    # model.add(Activation('tanh'))
    # model.add(AveragePooling1D(pool_size=2))
    # model.add(Conv1D(filters=10, kernel_size=2,
    #                  kernel_regularizer=regularizers.l1(0.01)))
    # model.add(Dropout(0.2))
    # model.add(Activation('tanh'))
    model.add(Flatten(input_shape=(12, InputDim)))
    model.add(Dense(1, kernel_regularizer=regularizers.l1(0.0002)))

    opt = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    if loadpath:
        model.load_weights(loadpath)
    print(model.summary())

    #训练模型
    checkpointer = ModelCheckpoint(filepath='ModelCNN/checkpoint-%i-val_mse_{val_mean_squared_error:.5f}.hdf5'%i,
                                   monitor='val_mean_squared_error', save_best_only=True)
    Xtrain_station = np.transpose(Xtrain[:, cand_list[i], :], (0, 2, 1))
    Xtest_station = np.transpose(Xtest[:, cand_list[i], :], (0, 2, 1))
    history = model.fit(Xtrain_station, Ytrain[:, i], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
              validation_data=(Xtest_station, Ytest[:, i]), callbacks=[checkpointer])

    if plot:
        plt.plot(history.history['mean_squared_error'], label='train')
        plt.plot(history.history['val_mean_squared_error'], label='test')
        plt.show()

    # 返回val_mse最小的模型
    least_mse = np.min(history.history['val_mean_squared_error'])
    model.load_weights('ModelCNN/checkpoint-%i-val_mse_%.5f.hdf5'%(i, least_mse))
    # 估计mse
    y1 = model.predict(np.transpose(Xtest[:, cand_list[i], :], (0, 2, 1)))
    # test数据集，计算预测结果
    y2 = model.predict(np.transpose(Vdata[:, cand_list[i], :], (0, 2, 1)))
    return model, y1, y2