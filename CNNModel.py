from keras.layers import*
from keras.models import*
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import pandas as pd
import os
from PreProcess import PreProcess


Model_dir = 'cnnModel/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)
mse_file = 'cnn_mse.csv'
submit = True

class TrafficCNN():
    def __init__(self):
        self.batchsize = 32
        self.epochs = 100
        self.time_step = 12
        
        self.input_dim = 0
        self.save_dir = ''
        
    def model(self, Xtrain, Ytrain, Xdev, Ydev, Xtest, loadpath=None):
        # 搭建网络
        model = Sequential()
        model.add(Conv1D(filters=self.input_dim, kernel_size=4, input_shape=(self.time_step, self.input_dim)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        #model.add(AveragePooling1D(pool_size=2))
        model.add(Conv1D(filters=int(self.input_dim*0.7), kernel_size=3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        #model.add(AveragePooling1D(pool_size=2))
        model.add(Conv1D(filters=int(self.input_dim*0.5), kernel_size=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1))
    
        opt = optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        if loadpath:
            model.load_weights(loadpath)
        #print(model.summary())
    
        #训练模型
        checkpointer = ModelCheckpoint(filepath=Model_dir + 'checkpoint-val_mse_{val_mean_squared_error:.5f}.hdf5',
                                       monitor='val_mean_squared_error', save_best_only=True)
        Xtrain = np.transpose(Xtrain, (0, 2, 1))
        Xdev = np.transpose(Xdev, (0, 2, 1))
        history = model.fit(Xtrain, Ytrain, epochs=self.epochs, batch_size=self.batchsize, verbose=2,
                  validation_data=(Xdev, Ydev), callbacks=[checkpointer])
    
        # 返回val_mse最小的模型
        least_mse = np.min(history.history['val_mean_squared_error'])
        print(least_mse)
        model.load_weights(Model_dir + 'checkpoint-val_mse_%.5f.hdf5'%(least_mse))
        # test数据集，计算预测结果
        Yhat = model.predict(np.transpose(Xtest, (0, 2, 1)))
        return model, least_mse, Yhat


if __name__ == '__main__':
    preprocess = PreProcess()
    cnn = TrafficCNN()
    cand_list = preprocess.select_nearest()
    Ysubmit = []
    mse_mat = []
    for predictday in [14, 17, 20]:
        Xtrain, Ytrain, Xdev, Ydev, Xtest = preprocess.readdata(predictday)
        mse_station = []
        Vy = []
        # 创建模型保存路径
        if not os.path.exists(Model_dir + 'predictday%i'%predictday):
            os.mkdir(Model_dir + 'predictday%i'%predictday)
        for i in range(228):
            # 模型预参数
            cnn.input_dim = len(cand_list[i])
            save_dir = Model_dir + 'predictday%i/station%i/'%(predictday, i)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cnn.save_dir = save_dir

            Xtrain0, Ytrain0 = Xtrain[:, cand_list[i], :], Ytrain[:, i]
            Xdev0, Ydev0 = Xdev[:, cand_list[i], :], Ydev[:, i]
            Xtest0 = Xtest[:, cand_list[i], :]

            #训练模型
            print('>start training model for station-%i predictday-%i'%(i, predictday))
            print('num of features:%i'%cnn.input_dim)
            _, mse, Yhat = cnn.model(Xtrain0, Ytrain0, Xdev0, Ydev0, Xtest0)
            mse_station.append(mse)
            Vy.append(Yhat)
            print('>training finished! final val_mse:%.5f'%mse)
        print('>>拟合所有%ith timepoint 模型完成！validation_mse: %f' % (predictday + 1, np.mean(mse_station)))
        print('每个站点的mse为:')
        print(mse_station)
        Ysubmit.append(np.array(Vy))
        mse_mat.append(mse_station)
    mse_mat = np.array(mse_mat)
    print('>>拟合所有模型完成！validation_mse: %f' % np.mean(mse_mat))
    mse_mat = pd.DataFrame(10000 * mse_mat.transpose(), columns=('3', '6', '9'))
    mse_mat.index = list(range(228))
    mse_mat = mse_mat.round(2)
    mse_mat.to_csv(mse_file, header=True, index=True)

    if submit:
        # 生成提交的csv文件
        Ysubmit = preprocess.data_inv_tf(np.array(Ysubmit))
        res = pd.DataFrame(columns=['Id', 'Expected'])
        timepoint = [15, 30, 45]
        for d in range(80):
            for t in range(3):
                for i in range(228):
                    res = res.append({'Id': '%i_%i_%i' % (d, timepoint[t], i), 'Expected': Ysubmit[t, i, d]},
                                     ignore_index=True)
        res.to_csv('prediction_cnn.csv', header=True, index=False)

