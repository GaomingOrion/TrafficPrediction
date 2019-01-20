import pandas as pd
import numpy as np
def cal_cov(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df1.columns = ['Id', 'Expected1']
    df2 = pd.read_csv(csv2)
    df2.columns = ['Id', 'Expected2']
    df_merge = pd.merge(df1, df2, how='inner')
    return np.mean(np.square(df_merge['Expected1']-df_merge['Expected2']))

# def ensemble(csv1, csv2, weight, outpath):
#     df1 = pd.read_csv(csv1)
#     df1.columns = ['Id', 'Expected1']
#     df2 = pd.read_csv(csv2)
#     df2.columns = ['Id', 'Expected2']
#     df_merge = pd.merge(df1, df2, how='inner')
#     df_merge['Expected'] = weight*df_merge['Expected1']+(1-weight)*df_merge['Expected2']
#     new_submit = df_merge[['Id', 'Expected']]
#     new_submit.to_csv(outpath, header=True, index=False)

def ensemble(csvlist, weight, outpath):
    df_merged, Expected_mean = 0, 0
    for i in range(len(csvlist)):
        df = pd.read_csv(csvlist[i])
        df.columns = ['Id', 'Expected'+str(i)]
        if i == 0:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, how='inner')
    for i in range(len(csvlist)):
        if i == 0:
            Expected_mean = weight[i]*df_merged['Expected'+str(i)]
        else:
            Expected_mean += weight[i]*df_merged['Expected'+str(i)]
    df_merged['Expected'] = Expected_mean
    new_submit = df_merged[['Id', 'Expected']]
    new_submit.to_csv(outpath, header=True, index=False)

if __name__ == '__main__':
    # csv_dir = 'csv_file/'
    # csvs = ['prediction_tnn.csv', 'submit_crnn.csv',
    #         'lstm.csv', 'bignet_crnn.csv',
    #         'lm_byinterval.csv', 'lm_reg_byinterval.csv', 'lasso.csv',
    #         'ensemble_ult.csv', 'ensemble_ult_ult.csv', 'ensemble3.csv']
    # weight = [0.1]*10
    # ensemble([csv_dir+x for x in csvs], weight, csv_dir+'ensemble_0000.csv')

    csv_dir = 'csv_file/'
    csvs = ['prediction_tnn.csv', 'submit_crnn.csv',
             'bignet_crnn.csv',
             'lasso.csv']


    for i in range(len(csvs)):
        for j in range(0, len(csvs)):
            print(csvs[i], csvs[j])
            print(cal_cov(csv_dir+csvs[i], csv_dir+csvs[j]))

