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


def main():
    df = np.array(pd.read_csv('data/train/1.csv', encoding='utf-8', names=list(range(228))))
    dis_mat = np.array(pd.read_csv('data/distance.csv', encoding='utf-8', names=list(range(228)))) / 1000
    coord = get_coord(dis_mat)
    a = adjust(coord)
    for i in range(228):
        x, y = a[i]
        v = ','.join(map(lambda x:str(np.round(x, 1)), df[[36*i for i in range(2, 6)], i]))
        label = str(i) + ',' + v
        #label = "%i,(%.2f,%.2f),"%(i, coord[i][0], coord[i][1])+v
        plt.scatter(x, y, s=1, label=label)
    datacursor(formatter='{label}'.format)
    plt.show()
    return a

if __name__ == '__main__':
    main()