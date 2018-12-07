import pandas as pd
import numpy as np

ts = []
for k in range(34):
    df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
    df = np.array(df).transpose()
    ts.append(df)
ts = np.array(ts)
ts_mean = np.mean(ts, axis=0)
