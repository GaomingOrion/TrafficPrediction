from keras.layers import*
from keras.models import*
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import pandas as pd

import matplotlib.pyplot as plt

from random import shuffle
from SeasonAnalyse import ts_mean

EPOCHS = 500
BATCH_SIZE = 64

InputDim = 228
Nsample = 63
Ntrain = 9000
Ntest = 600
def lstm():
    model = Sequential()
    model.add(LSTM(100, input_shape=(InputDim, 12), dropout=0.2))
    model.add(Dense(InputDim))
    print(model.summary())
    opt = optimizers.RMSprop(lr=0.0005)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

if __name__ == '__main__':
    Xdata = []
    Ydata = []
    for k in range(34):
        df = pd.read_csv('data/train/%i.csv' % k, encoding='utf-8', names=list(range(228)))
        df = np.array(df).transpose()[:, :Nsample]
        df = (df-ts_mean[:, :Nsample])*0.1
        Xdata += [df[:, i:(i+12)] for i in range(50)]
        Ydata += [df[:, i+12] for i in range(50)]
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    print(Xdata.shape, Ydata.shape)
    randsample = list(range(Xdata.shape[0]))
    shuffle(randsample)
    Xtrain = Xdata[randsample[:1500], :, :]
    Ytrain = Ydata[randsample[:1500], :]
    Xtest = Xdata[randsample[1500:], :, :]
    Ytest = Ydata[randsample[1500:], :]

    model = lstm()
    model.load_weights('models2/checkpoint-01e-val_mse_0.08126.hdf5')
    checkpointer = ModelCheckpoint(filepath='models2/checkpoint-{epoch:02d}e-val_mse_{val_mean_squared_error:.5f}.hdf5',
                                   monitor='val_mean_squared_error', save_best_only=True)
    history = model.fit(Xtrain, Ytrain, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
              validation_data=(Xtest, Ytest), callbacks=[checkpointer])
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.show()