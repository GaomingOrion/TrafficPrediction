import pandas as pd
import numpy as np
mse_list = []
def read_mse(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append(float(line.split('\t')[1]))
    return res

def ave_pre(csv1, csv2, outfile):
    f1 = pd.read_csv(csv1, header=0)
    f2 = pd.read_csv(csv2, header=0)
    f = pd.merge(f1, f2, on='Id', how='inner')
    f['Expected'] = (f['Expected_x']+f['Expected_y'])/2
    res = f[['Id', 'Expected']]
    res.to_csv(outfile, header=True, index=False)

if __name__ == '__main__':
    ave_pre('csv_file/prediction_tnn.csv', 'csv_file/submit_crnn.csv', 'csv_file/ensemble.csv')
    a = read_mse('csv_file/crnn_mse_1.txt')
    print(np.mean(a))