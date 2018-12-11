from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import os
from PreProcess import PreProcess


class TrafficCNN():
    def __init__(self):
        self.batchsize = 32
        self.epochs = 30
        self.time_step = 12

        self.input_dim = 228
        self.save_dir = ''

    def model(self, Xtrain, Ytrain, Xdev, Ydev, Xtest, loadpath=None):
        # 搭建网络
        model = Sequential()
        model.add(BatchNormalization(momentum=0.9, input_shape=(self.time_step, self.input_dim)))
        model.add(Conv1D(filters=200, kernel_size=4))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=100, kernel_size=1))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=200, kernel_size=3))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=100, kernel_size=1))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=200, kernel_size=3))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=100, kernel_size=1))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=200, kernel_size=2))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(228))

        opt = optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        if loadpath:
            model.load_weights(loadpath)
        print(model.summary())

        # 训练模型
        checkpointer = ModelCheckpoint(filepath=self.save_dir + 'checkpoint-val_mse_{val_mean_squared_error:.6f}.hdf5',
                                       monitor='val_mean_squared_error', save_best_only=True)
        Xtrain = np.transpose(Xtrain, (0, 2, 1))
        Xdev = np.transpose(Xdev, (0, 2, 1))
        history = model.fit(Xtrain, Ytrain, epochs=self.epochs, batch_size=self.batchsize, verbose=2,
                            validation_data=(Xdev, Ydev), callbacks=[checkpointer])

        # 返回val_mse最小的模型
        least_mse = np.min(history.history['val_mean_squared_error'])
        model.load_weights(self.save_dir + 'checkpoint-val_mse_%.6f.hdf5' % (least_mse))
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
    num_features = 228
    cand_list = preprocess.select_k_nearest(num_features)
    Model_dir = 'cnnModel_all/'
    if not os.path.exists(Model_dir):
        os.mkdir(Model_dir)

    for predictday in [14, 17, 20]:
        Xtrain, Ytrain, Xdev, Ydev, Xtest = preprocess.readdata(predictday)
        # 创建模型保存路径
        if not os.path.exists(Model_dir + 'predictday%i' % predictday):
            os.mkdir(Model_dir + 'predictday%i' % predictday)
        cnn.save_dir = Model_dir + 'predictday%i/'%predictday
        mse, Yhat = cnn.model(Xtrain, Ytrain, Xdev, Ydev, Xtest)
        Yhat = preprocess.data_inv_tf(Yhat)
        print('>训练结束，val_mse: %.6f'%mse)

        timepoint = {14:15, 17:30, 20:45}[predictday]
        with open('csv_file/submit_largecnn.csv', 'a') as f:
            for i in range(80):
                for j in range(228):
                    f.write(str(i) + '_' + str(timepoint) + '_' + str(j) + ',' + str(Yhat[i, j]) + '\n')
        print('>写入结果完成')

