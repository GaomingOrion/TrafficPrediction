from keras.layers import*
from keras.models import*
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import os
from PreProcess import PreProcess

class TrafficCNN():
    def __init__(self):
        self.batchsize = 32
        self.epochs = 120
        self.time_step = 12
        
        self.input_dim = 228
        self.save_dir = ''
        
    def model(self, Xtrain, Ytrain, Xdev, Ydev, Xtest, loadpath=None):
        # 搭建网络
        model = Sequential()
        model.add(BatchNormalization(momentum=0.9, input_shape=(self.time_step, self.input_dim)))
        model.add(Conv1D(filters=100, kernel_size=4))
        model.add(Activation('relu'))
        # model.add(Conv1D(filters=100, kernel_size=3))
        # model.add(Activation('relu'))
        model.add(Conv1D(filters=50, kernel_size=3))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1, activity_regularizer=regularizers.l2(1e-3)))
    
        opt = optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        if loadpath:
            model.load_weights(loadpath)
        print(model.summary())
    
        #训练模型
        checkpointer = ModelCheckpoint(filepath=self.save_dir + 'checkpoint-val_mse_{val_mean_squared_error:.6f}.hdf5',
                                       monitor='val_mean_squared_error', save_best_only=True)
        Xtrain = np.transpose(Xtrain, (0, 2, 1))
        Xdev = np.transpose(Xdev, (0, 2, 1))
        history = model.fit(Xtrain, Ytrain, epochs=self.epochs, batch_size=self.batchsize, verbose=2,
                  validation_data=(Xdev, Ydev), callbacks=[checkpointer])
    
        # 返回val_mse最小的模型
        least_mse = np.min(history.history['val_mean_squared_error'])
        model.load_weights(self.save_dir + 'checkpoint-val_mse_%.6f.hdf5'%(least_mse))
        # test数据集，计算预测结果
        Yhat = model.predict(np.transpose(Xtest, (0, 2, 1)))
        del model
        return least_mse, Yhat


if __name__ == '__main__':
    # tensorflow use cpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    preprocess = PreProcess()
    cnn = TrafficCNN()
    #num_features = 30
    cand_list = preprocess.select_nearest()
    Model_dir = 'cnnModel/'
    if not os.path.exists(Model_dir):
        os.mkdir(Model_dir)
    test_file = 'csv_file/test.txt'
    # with open(test_file, 'a') as f:
    #     f.write(str(num_features) + '\n')

    for predictday in [14, 17, 20]:
        Xtrain, Ytrain, Xdev, Ydev, Xtest = preprocess.readdata(predictday)
        # 创建模型保存路径
        if not os.path.exists(Model_dir + 'predictday%i'%predictday):
            os.mkdir(Model_dir + 'predictday%i'%predictday)
        for i in [0, 26, 31]:
            # 模型预参数
            cnn.input_dim = len(cand_list[i])
            save_dir = Model_dir + 'predictday%i/station%i/'%(predictday, i)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cnn.save_dir = save_dir

            Xtrain0, Ytrain0 = Xtrain[:, cand_list[i], :], Ytrain[:, i]
            Xdev0, Ydev0 = Xdev[:, cand_list[i], :], Ydev[:, i]
            Xtest0 = Xtest[:, cand_list[i], :]

            # 训练模型
            print('>start training model for station-%i predictday-%i'%(i, predictday))
            print('num of features:%i'%cnn.input_dim)
            mse, Yhat = cnn.model(Xtrain0, Ytrain0, Xdev0, Ydev0, Xtest0)
            print('>training finished! final val_mse:%.5f'%mse)

            # 写入mse
            with open(test_file, 'a') as f:
                f.write(str(predictday) + ',' + str(i) + '\t' + str(np.round(10000*mse, 2)) + '\n')

