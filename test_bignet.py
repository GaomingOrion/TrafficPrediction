import tensorflow as tf
import numpy as np

from PreProcess import PreProcess
from CRNNmodel import TrafficCRNN
import os

class bigNet():
    def __init__(self):
        self.batchsize = 32
        self.epochs = 40
        self.time_step = 12
        # will change
        self.input_dim = 0
        self.save_dir = ''
        self.best_model = ''
        self.val_mse_min = np.inf

    def BuildModel(self):
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.time_step))
        self.W_ph = tf.placeholder(tf.float32, shape=(None, self.input_dim, 1))
        self.y_ph = tf.placeholder(tf.float32, shape=(None,))
        self.X_norm = tf.layers.batch_normalization(self.X_ph, momentum=0.8)
        X_in = tf.multiply(self.W_ph, self.X_norm)
        Xconv = tf.layers.conv1d(inputs=tf.transpose(X_in, perm=[0, 2, 1]),
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
        _, rnn_out = tf.nn.dynamic_rnn(cell2, rnn_in2, dtype=tf.float32,scope='second')

        self.y_pred = tf.layers.dense(rnn_out.h, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.total_loss = tf.reduce_mean(tf.squared_difference(tf.reshape(self.y_ph, (-1, 1)), self.y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.total_loss)

    def Predict(self, sess, Xdev, Wdev):
        per_cal = 200
        y_hat = np.array([])
        low = 0
        while True:
            n_sample = min(per_cal, Xdev.shape[0]-low)
            feed_dict = {self.X_ph: Xdev[low:low+n_sample],
                         self.W_ph: Wdev[low:low+n_sample]}
            y_percal = sess.run(self.y_pred, feed_dict=feed_dict)
            y_percal = y_percal.reshape(-1)
            y_hat = np.concatenate((y_hat, y_percal))
            low += per_cal
            if low >= Xdev.shape[0]:
                break
        return y_hat

    def Evaluate(self, sess, Xdev, Wdev, Ydev):
        y_hat = self.Predict(sess, Xdev, Wdev)
        mse = np.mean(np.square(Ydev-y_hat))
        return y_hat, mse

    def Test(self, Xtest, Wtest):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.best_model)
            y_test = self.Predict(sess, Xtest, Wtest)
        return y_test

    def TrainModel(self, Xtrain, Wtrain, Ytrain, Xdev, Wdev, Ydev, previous_modelpath=None, epochsep=1):
        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            if previous_modelpath:
                saver.restore(sess, previous_modelpath)
            else:
                sess.run(tf.global_variables_initializer())

            low = 0
            for epoch_idx in range(self.epochs):
                epoch = epochsep*(epoch_idx+1)
                shuffle_idx = list(range(Xtrain.shape[0]))
                np.random.shuffle(shuffle_idx)
                while True:
                    n_sample = min(self.batchsize, Xtrain.shape[0]-low)
                    feed_dict = {self.X_ph: Xtrain[shuffle_idx[low:low+n_sample]],
                                 self.W_ph: Wtrain[shuffle_idx[low:low+n_sample]],
                                 self.y_ph: Ytrain[shuffle_idx[low:low+n_sample]]}
                    sess.run(self.train_op, feed_dict=feed_dict)
                    low += n_sample
                    if low >= epochsep*Xtrain.shape[0]:
                        low = 0
                        _, loss = self.Evaluate(sess, Xtrain, Wtrain, Ytrain)
                        _, mse = self.Evaluate(sess, Xdev, Wdev, Ydev)
                        model_path = self.save_dir+'model-e%i-loss%.4f-val_mse-%.4f'%(epoch, loss, mse)
                        if mse < self.val_mse_min:
                            self.best_model = model_path
                            self.val_mse_min = mse
                            saver.save(sess, model_path)
                        print('epoch%.1f--loss:%.4f, val_mse:%.4f'%(epoch, loss, mse))
                        break
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
    Xtrain, Wtrain, Ytrain, Xdev, Wdev, Ydev, Xtest, Wtest = p.generate_data4bignet(max([0,36 * i-2]), 36*i+3, predictday, 60, return_weight=True)
    model.input_dim = Xtrain.shape[1]
    model.save_dir = Model_dir + 'predictday%i/%i/' % (predictday, i)
    model.val_mse_min = np.inf

    # 训练模型
    print('>start training model for predictday-%i start-%i' % (predictday, i))
    print('num of features:%i' % model.input_dim)
    model.BuildModel()
    model.TrainModel(Xtrain, Wtrain, Ytrain, Xdev, Wdev, Ydev, previous_modelpath=None, epochsep=1)
    mse = model.val_mse_min
    Yhat = model.Test(Xtest, Wtest)

    tf.reset_default_graph()
    Yhat = p.data_inv_tf(Yhat)
    print('>training finished! final val_mse:%.5f' % mse)

    # 写入结果文件
    with open(mse_file, 'a') as f:
        f.write(str(predictday) + ',' + str(i) + '\t' + str(round(10000*mse, 2)) + '\n')
    # timepoint = {14:15, 17:30, 20:45}[predictday]
    # with open(submit_file, 'a') as f:
    #     for k in range(228):
    #         for j in range(10):
    #             idx = 10*k + j
    #             file_idx = 10*j+i
    #             f.write(str(file_idx) + '_' + str(timepoint) + '_' + str(i) + ',' + str(Yhat[idx]) + '\n')
    # print('>结果写入完成')
    return mse

if __name__ == '__main__':
    for predictday in [20]:
        mse = []
        for i in [2]:
            mse.append(main(predictday, i))
        print(np.mean(mse))



