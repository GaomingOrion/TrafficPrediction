#import pandas as pd
import numpy as np
mse_list = []
def read_mse(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append(float(line.split('\t')[1]))
    return res

if __name__ == '__main__':
    a = read_mse('csv_file/crnn_mse_3.txt')
    print(np.mean(a))