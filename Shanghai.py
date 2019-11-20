# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:52:29 2019

@author: sjawo
"""

from keras.callbacks import EarlyStopping
from math import sqrt
from keras.metrics import mse, mae, mape, msle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import h5py
from keras.layers import Dense, Activation, LSTM  
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")
from statistics import mean
np.random.seed(7)


np.random.seed(7)

shanghai = pd.read_csv('ShanghaiPM20100101_20151231.csv', header=0,
                       names=['No', 'year', 'month', 'day', 'hour', 'season', 'PM_Jingan', 'PM_US Post','PM_Xuhui',
                             'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws',
                             'precipitation', 'Iprec'],
                       dtype={'No': int, 'year': int, 'month': int, 'day': int, 'hour': int, 'season': str,
                              'PM_Jingan': str, 'PM_US Post': str,'PM_Xuhui': str,
                             'DEWP': float, 'HUMI': float, 'PRES': float, 'TEMP': float, 'cbwd': str, 'Iws': float,
                             'precipitation': float, 'Iprec': float})

shanghai.set_index('No', drop=True)
shanghai = shanghai[(shanghai.No >= 27829)]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 6]):
        shanghai.iloc[x, 6] = shanghai.iloc[x - 1, 6]
shanghai['PM_Jingan'] = shanghai['PM_Jingan'].astype(float)

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 7]):
        shanghai.iloc[x, 7] = shanghai.iloc[x - 1, 7]
shanghai['PM_US Post'] = shanghai['PM_US Post'].astype(float)

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 8]):
        shanghai.iloc[x, 8] = shanghai.iloc[x - 1, 8]
shanghai['PM_Xuhui'] = shanghai['PM_Xuhui'].astype(float)


for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 13]):
        shanghai.iloc[x, 13] = shanghai.iloc[x - 1, 13]


for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 15]):
        shanghai.iloc[x, 15] = shanghai.iloc[x - 1, 15]
        
for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 16]):
        shanghai.iloc[x, 16] = shanghai.iloc[x - 1, 16]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 9]):
        shanghai.iloc[x, 9] = shanghai.iloc[x - 1, 9]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 10]):
        shanghai.iloc[x, 10] = shanghai.iloc[x - 1, 10]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 11]):
        shanghai.iloc[x, 11] = shanghai.iloc[x - 1, 11]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 12]):
        shanghai.iloc[x, 12] = shanghai.iloc[x - 1, 12]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 14]):
        shanghai.iloc[x, 14] = shanghai.iloc[x - 1, 14]

for x in range(0, shanghai.shape[0]):
    if pd.isna(shanghai.iloc[x, 16]):
        shanghai.iloc[x, 16] = shanghai.iloc[x - 1, 16]
shanghai['season'] = shanghai['season'].astype(int)

for x in range(0, shanghai.shape[0]):
    if shanghai.iloc[x, 13] == 'NE':
        shanghai.iloc[x, 13] = 1
    if shanghai.iloc[x, 13] == 'SE':
        shanghai.iloc[x, 13] = 1
    if shanghai.iloc[x, 13] == 'cv':
        shanghai.iloc[x, 13] = 3
    if shanghai.iloc[x, 13] == 'NW':
        shanghai.iloc[x, 13] = 4
    if shanghai.iloc[x, 13] == 'SW':
        shanghai.iloc[x, 13] = 3

shanghai['cbwd'] = shanghai['cbwd'].astype(float)


#pozostałe stacje - na końcu 7 lub 8
dane = shanghai.iloc[:, [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 8]]
X = shanghai.iloc[:, [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 8]]

# scaling data
# =============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
# saving scaler
# scaler_filename = "scaler_hourly"
scaled = scaler.fit_transform(X)
X_scaled = scaled
size = X_scaled.shape[1]
Y = scaled[:, size - 1]
X = scaled[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
size = X.shape[1]

# building model
# =============================================================================
# train-test split and reshaping to 3D
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, shuffle=True)
X_train = np.reshape(X_train, (-1, 1, size))
X_test = np.reshape(X_test, (-1, 1, size))

# NEURAL NETWORK
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))

model.add(Dense(32, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
early_stopping = EarlyStopping(monitor ='val_loss', mode='min', verbose=1, patience=100)
# training
history = model.fit(X_train, y_train, epochs=150, batch_size=25, validation_split=0.1, verbose=2,
                    callbacks=[early_stopping])
# predictions for test data
predictions = model.predict(X_test)

plt.figure(0)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# reshaping and inversing normalization
# =============================================================================
prediction_reshaped = np.zeros((len(predictions), size + 1))
testY_reshaped = np.zeros((len(y_test), size + 1))

prediction_r = np.reshape(predictions, (len(predictions),))
testY_r = np.reshape(y_test, (len(y_test),))

prediction_reshaped[:, size] = prediction_r
testY_reshaped[:, size] = testY_r

prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:, size]
testY_inversed = scaler.inverse_transform(testY_reshaped)[:, size]

# calculating error rates
# =============================================================================
msee = mean_squared_error(testY_inversed, prediction_inversed)
rmse = sqrt(mean_squared_error(testY_inversed, prediction_inversed))
maae = mean_absolute_error(testY_inversed, prediction_inversed)
r2 = r2_score(testY_inversed, prediction_inversed)

plt.figure(0)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'red')
plt.plot(history.history['val_loss'], 'blue')
plt.title('Model strat')
plt.ylabel('funkcja strat')
plt.xlabel('epoka')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.subplot(1, 2, 2)
plt.plot(prediction_inversed[1250:1350], 'red', label='predykcja')
plt.plot(testY_inversed[1250:1350], 'blue', label='rzeczywiste dane')
plt.title('Predykcja vs rzeczywiste dane')
plt.xlabel('próbki')
plt.ylabel('PM2.5')
plt.legend(loc='upper right')
plt.show()

# print error rates and R^2
# =============================================================================
print('MSE: %.3f' % msee)
print('RMSE: %.3f' % rmse)
print('MAE: %.3f' % maae)
print('R^2: %.3f' % r2)
# =============================================================================