import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf

dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228))))

# 选出每个station距离小于5km的其他station
cand_list = [[k] for k in range(228)]
for i in range(228):
    for j in range(i+1, 228):
        if dis_mat[i, j] < 5000:
            cand_list[i].append(j)
            cand_list[j].append(i)


ts = []
for k in range(34):
    df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
    df = np.array(df).transpose()
    ts.append(df)
ts = np.array(ts)
ts_mean = np.mean(ts, axis=0)

for i in range(1):
    for j in cand_list[i]:
        print((i, j))
        for k in range(34):
            print('for %ith file:'%k, ccf(ts[k, i, :], ts[k, j, :])[:12])