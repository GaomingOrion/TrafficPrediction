import tensorflow as tf
import numpy as np
import pandas as pd

from PreProcess import PreProcess

import os
# tensorflow use cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

Model_dir = 'tnnModel/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)
mse_file = 'tnn_mse.csv'
class TrafficNN():
    def __init__(self):
        self.batchsize = 32
        self.epoches = 100
        self.time_step = 12
        # will change
        self.input_dim = 0
        self.save_dir = ''
        self.best_model = ''
        self.val_mse_min = np.inf
        self.dense_num = 1

    def BuildModel(self):
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.time_step))
        self.y_ph = tf.placeholder(tf.float32, shape=(None,))
        A_matrix = tf.get_variable('A_matrix', shape=[self.input_dim, self.time_step, self.time_step])
        X_left = tf.gather(self.X_ph, [0]*self.input_dim, axis=1)
        X_left = tf.reshape(X_left, shape=(-1, self.input_dim, 1, self.time_step))
        X_right = tf.reshape(self.X_ph, shape=(-1, self.input_dim, self.time_step, 1))

        Dense = tf.reshape(tf.einsum('ntij,tjk,ntkl->ntil', X_left, A_matrix, X_right),
                              shape=(-1, self.input_dim))
        Dense = tf.nn.sigmoid(Dense)
        Dense = tf.layers.dense(Dense, self.input_dim)
        Dense = tf.nn.sigmoid(Dense)
        if self.dense_num == 2:
            Dense = tf.layers.dense(Dense, 20)
            Dense = tf.nn.sigmoid(Dense)
            Dense = tf.layers.dense(Dense, 10)
            Dense = tf.nn.sigmoid(Dense)
        self.y_pred = tf.layers.dense(Dense, 1, kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(tf.squared_difference(tf.reshape(self.y_ph, (-1, 1)), self.y_pred))\
                    +reg_losses
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)


    def Predict(self, sess, Xdev):
        per_cal = 200
        y_hat = np.array([])
        low = 0
        while True:
            n_sample = min(per_cal, Xdev.shape[0]-low)
            feed_dict = {self.X_ph: Xdev[low:low+n_sample]}
            y_percal = sess.run(self.y_pred, feed_dict=feed_dict)
            y_percal = y_percal.reshape(-1)
            y_hat = np.concatenate((y_hat, y_percal))
            low += per_cal
            if low >= Xdev.shape[0]:
                break
        return y_hat

    def Evaluate(self, sess, Xdev, Ydev):
        y_hat = self.Predict(sess, Xdev)
        mse = np.mean(np.square(Ydev-y_hat))
        return y_hat, mse

    def Test(self, Xtest):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.best_model)
            y_test = self.Predict(sess, Xtest)
        return y_test

    def TrainModel(self, Xtrain, Ytrain, Xdev, Ydev, previous_modelpath=None):
        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            if previous_modelpath:
                saver.restore(sess, previous_modelpath)
            else:
                sess.run(tf.global_variables_initializer())

            low = 0
            for epoch in range(1, self.epoches+1):
                shuffle_idx = list(range(Xtrain.shape[0]))
                np.random.shuffle(shuffle_idx)
                while True:
                    n_sample = min(self.batchsize, Xtrain.shape[0]-low)
                    feed_dict = {self.X_ph: Xtrain[shuffle_idx[low:low+n_sample]],
                                 self.y_ph: Ytrain[shuffle_idx[low:low+n_sample]]}
                    sess.run(self.train_op, feed_dict=feed_dict)
                    low += n_sample
                    if low >= Xtrain.shape[0]:
                        low = 0
                        _, loss = self.Evaluate(sess, Xtrain, Ytrain)
                        _, mse = self.Evaluate(sess, Xdev, Ydev)
                        model_path = self.save_dir+'model-e%i-loss%.4f-val_mse-%.4f'%(epoch, loss, mse)
                        if mse < self.val_mse_min:
                            self.best_model = model_path
                            self.val_mse_min = mse
                            saver.save(sess, model_path)
                        print('epoch%i--loss:%.4f, val_mse:%.4f'%(epoch, loss, mse))
                        break



if __name__ == '__main__':
    submit = False
    tnn = TrafficNN()
    preprocess = PreProcess()
    cand_list = preprocess.select_nearest()
    # 保存要提交的结果
    Ysubmit = []
    # 保存所有模型的mse
    mse_mat = []
    for predictday in [14, 17, 20]:
        Xtrain, Ytrain, Xdev, Ydev, Xtest = preprocess.readdata(predictday)
        mse_station = []
        Vy = []
        # 模型保存路径
        if not os.path.exists(Model_dir + 'predictday%i'%predictday):
            os.mkdir(Model_dir + 'predictday%i'%predictday)

        for i in range(228):
        #for i in [6]:
            # 模型预参数
            tnn.input_dim = len(cand_list[i])
            tnn.val_mse_min = np.inf
            tnn.dense_num = 1 if predictday==14 else 2
            save_dir = Model_dir + 'predictday%i/station%i/'%(predictday, i)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            tnn.save_dir = save_dir

            # 拟合模型
            Xtrain0, Ytrain0 = Xtrain[:, cand_list[i], :], Ytrain[:, i]
            Xdev0, Ydev0 = Xdev[:, cand_list[i], :], Ydev[:, i]
            Xtest0 = Xtest[:, cand_list[i], :]
            print('>>start training model for station-%i predictday-%i'%(i, predictday))
            print('num of features:%i'%tnn.input_dim)
            tnn.BuildModel()
            tnn.TrainModel(Xtrain0, Ytrain0, Xdev0, Ydev0)

            mse_station.append(tnn.val_mse_min)
            Vy.append(tnn.Test(Xtest0))

            tf.reset_default_graph()
            print('>>training finished! final val_mse:%.5f'%tnn.val_mse_min)
        print('>>>拟合所有%ith timepoint 模型完成！validation_mse: %f'%(predictday+1, np.mean(mse_station)))
        print('每个站点的mse为:')
        print(mse_station)
        Ysubmit.append(np.array(Vy))
        mse_mat.append(mse_station)
    mse_mat = np.array(mse_mat)
    print('>>>拟合所有模型完成！validation_mse: %f'%np.mean(mse_mat))
    mse_mat = pd.DataFrame(10000*mse_mat.transpose(), columns=('3', '6', '9'))
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
        res.to_csv('prediction.csv', header=True, index=False)




