import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

for k in range(34):
    os.mkdir('tsplot/%i'%k)
    df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
    df = np.array(df)
    for i in range(228):
        plt.plot(df[:, i ])
        plt.savefig('tsplot/%i/%i.png'%(k, i))
        plt.close()