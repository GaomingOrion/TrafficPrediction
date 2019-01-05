import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# for k in range(34):
#     os.mkdir('tsplot/%i'%k)
#     df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
#     df = np.array(df)
#     for i in range(228):
#         plt.plot(df[:, i ])
#         plt.savefig('tsplot/%i/%i.png'%(k, i))
#         plt.close()

ts = []
for k in range(80):
    df = pd.read_csv('data/test/%i.csv' % k, encoding='utf-8', names=list(range(228)))
    df = np.array(df)
    ts.append(df)
ts = np.array(ts)
os.mkdir('comparePlot')
for i in range(228):
    os.mkdir('comparePlot/%i'%i)
    for j in range(8):
        plt.ylim(0, 90)
        for k in range(10):
            plt.plot(list(range(36*j, 36*j+12)), ts[8*k+j, :, i])
        plt.savefig('comparePlot/%i/%i.png'%(i, 36*j))
        plt.close()

ts = []
for k in range(34):
    df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
    df = np.array(df)
    ts.append(df)
ts = np.array(ts)
for i in range(228):
    for j in range(8):
        plt.ylim(0, 90)
        for k in range(34):
            plt.plot(list(range(36*j, 36*j+12)), ts[k, 36*j:(36*j+12), i])
        plt.savefig('comparePlot/%i/%i_train.png'%(i, 36*j))
        plt.close()