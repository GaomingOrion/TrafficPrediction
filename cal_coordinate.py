import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpldatacursor import datacursor

def l2_dis(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def get_coord(dis_mat):
    a = []
    a.append([0, 0])
    a.append([dis_mat[1, 0], 0])
    x1 = dis_mat[1, 0]
    for i in range(2, 228):
        x = (x1 ** 2 + dis_mat[i, 0] ** 2 - dis_mat[i, 1] ** 2) / (2 * x1)
        y1 = np.sqrt(dis_mat[i, 0] ** 2 - x ** 2)
        y2 = -y1
        if i == 2:
            y = y1
        else:
            if abs(l2_dis([x, y1], a[2])-dis_mat[i,2])< abs(l2_dis([x, y2], a[2])-dis_mat[i,2]):
                y = y1
            else:
                y = y2
        a.append(([x, y]))
    return a


def adjust(a, tol=0.05):
    n = len(a)
    res = a[::]
    for i in range(n):
        for j in range(i+1, n):
            if l2_dis(a[i], a[j]) < tol:
                res[j] = [a[i][0]+0.1, a[i][1]+0.1]
    return res

def getmeanv():
    res = []
    for k in range(34):
        df = np.array(pd.read_csv('data/train/1.csv', encoding='utf-8', names=list(range(228))))
        res.append(df)
    res = np.array(res)
    return np.mean(res, axis=0)

def readmse():
    mse = pd.read_csv('csv_file/tnn_mse_1209.csv', index_col=0)
    return mse['9']

def jam_type():
    v = getmeanv()
    v_mean = np.mean(v.reshape(-1, 6, 228), axis=1)
    a = np.min(v_mean[10:18, :], axis=0)
    b = np.min(v_mean[32:38, :], axis=0)
    res = []
    for i in range(228):
        resi = ''
        if a[i] < 35:
            resi += '0,'
        # elif 30<= a[i] < 50:
        #     resi +='1,'
        else:
            resi += '1,'

        if b[i] < 35:
            resi += '0'
        # elif 30<= b[i] < 50:
        #     resi += '1'
        else:
            resi += '1'
        res.append(resi)
    return res

def main():
    df = getmeanv()
    mse = readmse()
    dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228)))) / 1000
    coord = get_coord(dis_mat)
    a = adjust(coord)
    jam = jam_type()
    color = ['r', 'black']
    for i in range(228):
        x, y = a[i]
        t = int(jam[i][-1])
        #v = ','.join(map(lambda x: str(int(x)), df[[i for i in range(6*12, 8*12+13, 2)], i]))
        #label = str(i) + ',' + v
        label = "%i,(%.2f,%.2f)"%(i, coord[i][0], coord[i][1]) +"," + str(int(mse[i]))
        plt.scatter(x, y, s=mse[i]/10, label=label, color=color[t])
    datacursor(formatter='{label}'.format)
    plt.show()
    return a

if __name__ == '__main__':
    main()