import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import yfinance as yf
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
class LSTMModel():
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dense(25, activation='linear'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    def forward(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        
    def predict(self, x):
        return self.model.predict(x)