import tensorflow as tf
import numpy as np

from PreProcess import PreProcess
from CRNNmodel import TrafficCRNN
import os

class bigNet(TrafficCRNN):
    def BuildModel(self):
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.time_step))
        self.y_ph = tf.placeholder(tf.float32, shape=(None,))
        self.X_norm = tf.layers.batch_normalization(self.X_ph, momentum=0.8)
        Xconv = tf.layers.conv1d(inputs=tf.transpose(self.X_norm, perm=[0, 2, 1]),
                        filters=self.input_dim, kernel_size=3,
                    kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu)
        # Xconv = tf.layers.conv1d(inputs=tf.transpose(Xconv, perm=[0, 2, 1]),
        #                 filters=self.input_dim+20, kernel_size=2,
        #             kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu)

        rnn_in = tf.transpose(Xconv, perm=[0, 2, 1])
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.input_dim, initializer=tf.orthogonal_initializer())
        #cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.5)
        #_, rnn_out = tf.nn.dynamic_rnn(cell, rnn_in, dtype=tf.float32)
        rnn_in2, _ = tf.nn.dynamic_rnn(cell, rnn_in, dtype=tf.float32, scope='first')
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=self.input_dim, initializer=tf.orthogonal_initializer())
        #cell2 = tf.contrib.rnn.DropoutWrapper(cell2, input_keep_prob=1.0, output_keep_prob=0.5)
        _, rnn_out = tf.nn.dynamic_rnn(cell2, rnn_in2, dtype=tf.float32, scope='second')

        self.y_pred = tf.layers.dense(rnn_out.h, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.total_loss = tf.reduce_mean(tf.squared_difference(tf.reshape(self.y_ph, (-1, 1)), self.y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.total_loss)

use_cpu = True
if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 模型保存路径
Model_dir = 'bigNet/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)
for predictday in [14, 17, 20]:
    if not os.path.exists(Model_dir + 'predictday%i/'%predictday):
        os.mkdir(Model_dir + 'predictday%i/'%predictday)
    for i in range(8):
        if not os.path.exists(Model_dir + 'predictday%i/%i/' % (predictday, i)):
            os.mkdir(Model_dir + 'predictday%i/%i/' % (predictday, i))

# 结果文件路径
mse_file = 'csv_file/bignet_mse.txt'
submit_file = 'csv_file/bignet_crnn.csv'

#其他
model = bigNet()
model.epochs = 40
p = PreProcess()

def main(predictday, i):
    Xtrain, Ytrain, Xdev, Ydev, Xtest = p.generate_data4bignet(max([0, 36 * i-2]), 36*i+3, predictday, 60)
    model.input_dim = Xtrain.shape[1]
    model.save_dir = Model_dir + 'predictday%i/%i/' % (predictday, i)
    model.val_mse_min = np.inf

    # 训练模型
    print('>start training model for predictday-%i start-%i' % (predictday, i))
    print('num of features:%i' % model.input_dim)
    model.BuildModel()
    model.TrainModel(Xtrain, Ytrain, Xdev, Ydev,
        previous_modelpath=None, epochsep=1)
    mse = model.val_mse_min
    Yhat = model.Test(Xtest)

    tf.reset_default_graph()
    Yhat = p.data_inv_tf(Yhat)
    print('>training finished! final val_mse:%.5f' % mse)

    # 写入结果文件
    with open(mse_file, 'a') as f:
        f.write(str(predictday) + ',' + str(i) + '\t' + str(round(10000*mse, 2)) + '\n')
    timepoint = {14:15, 17:30, 20:45}[predictday]
    with open(submit_file, 'a') as f:
        for k in range(228):
            for j in range(10):
                idx = 10*k + j
                file_idx = 10*j+i
                f.write(str(file_idx) + '_' + str(timepoint) + '_' + str(i) + ',' + str(Yhat[idx]) + '\n')
    print('>结果写入完成')
    return mse

if __name__ == '__main__':
    for predictday in [14]:
        mselist = []
        Yhat_list = []
        for i in range(8):
            mse, Yhat = main(predictday, i)
            mselist.append(mse)
            Yhat_list.append(Yhat)
        print(np.mean(mselist))
        Yhat_list = np.array(Yhat_list)
        np.save('csv_file/backup%i.npy' % predictday, Yhat_list)



