import tensorflow as tf
import numpy as np

from PreProcess import PreProcess
import os

class TrafficCRNN():
    def __init__(self):
        self.batchsize = 32
        self.epochs = 40
        self.time_step = 12
        # will change
        self.input_dim = 50
        self.save_dir = ''
        self.best_model = ''
        self.val_mse_min = np.inf

    def BuildModel(self):
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.time_step))
        self.y_ph = tf.placeholder(tf.float32, shape=(None,))
        self.X_norm = tf.layers.batch_normalization(self.X_ph, momentum=0.8)
        Xconv = tf.layers.conv1d(inputs=tf.transpose(self.X_norm, perm=[0, 2, 1]),
                        filters=self.input_dim+20, kernel_size=3,
                    kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu)
        # Xconv = tf.layers.conv1d(inputs=Xconv,
        #                 filters=self.input_dim+20, kernel_size=3,
        #             kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu)
        rnn_in = tf.transpose(Xconv, perm=[0, 2, 1])
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.input_dim+20, initializer=tf.orthogonal_initializer())
        _, rnn_out = tf.nn.dynamic_rnn(cell, rnn_in, dtype=tf.float32)

        self.y_pred = tf.layers.dense(rnn_out.h, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.total_loss = tf.reduce_mean(tf.squared_difference(tf.reshape(self.y_ph, (-1, 1)), self.y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)


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
            for epoch in range(1, self.epochs+1):
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
    p = PreProcess()
    Xtrain, Ytrain, Xdev, Ydev, Xtest = p.generate_data4bignet(36*3, 14, 50)
    model = TrafficCRNN()
    model.save_dir = 'bigNet/'
    model.BuildModel()
    model.TrainModel(Xtrain, Ytrain, Xdev, Ydev)



