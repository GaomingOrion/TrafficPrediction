from PreProcess import PreProcess
from CRNNmodel import TrafficCRNN
import tensorflow as tf
import numpy as np
import os

use_cpu = True
if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 模型保存路径
Model_dir = 'crnnModel/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)
for predictday in [14, 17, 20]:
    if not os.path.exists(Model_dir + 'predictday%i'%predictday):
        os.mkdir(Model_dir + 'predictday%i'%predictday)

# 结果文件路径
mse_file = 'csv_file/crnn_mse_1.txt'
submit_file = 'csv_file/submit_crnn_1.csv'

# 训练数据等
preprocess = PreProcess()
model = TrafficCRNN()
cand_list = preprocess.select_nearest()
Train_data = dict()
for predictday in [14, 17, 20]:
    Train_data[predictday] = preprocess.readdata(predictday)

def main(predictday, i):
    Xtrain, Ytrain, Xdev, Ydev, Xtest = Train_data[predictday]
    Xtrain, Ytrain = Xtrain[:, cand_list[i], :], Ytrain[:, i]
    Xdev, Ydev = Xdev[:, cand_list[i], :], Ydev[:, i]
    Xtest = Xtest[:, cand_list[i], :]

    # 模型参数
    model.input_dim = len(cand_list[i])
    model.val_mse_min = np.inf
    save_dir = Model_dir + 'predictday%i/station%i/' % (predictday, i)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model.save_dir = save_dir

    # 训练模型
    print('>start training model for station-%i predictday-%i' % (i, predictday))
    print('num of features:%i' % model.input_dim)
    model.BuildModel()
    model.TrainModel(Xtrain, Ytrain, Xdev, Ydev)
    mse = model.val_mse_min
    Yhat = model.Test(Xtest)

    tf.reset_default_graph()
    Yhat = preprocess.data_inv_tf(Yhat)
    print('>training finished! final val_mse:%.5f' % mse)

    # 写入结果文件
    with open(mse_file, 'a') as f:
        f.write(str(predictday) + ',' + str(i) + '\t' + str(round(10000*mse, 2)) + '\n')
    timepoint = {14:15, 17:30, 20:45}[predictday]
    with open(submit_file, 'a') as f:
        for k in range(Yhat.shape[0]):
            f.write(str(k) + '_' + str(timepoint) + '_' + str(i) + ',' + str(Yhat[k]) + '\n')
    print('>结果写入完成')

if __name__ == '__main__':
    predictday = 14
    for i in range(6, 228):
        main(predictday, i)
        # try:
        #     main(predictday, i)
        # except:
        #     with open('csv_file/error.txt', 'a+') as f:
        #         f.write(str(i)+'\n')
        #     tf.reset_default_graph()