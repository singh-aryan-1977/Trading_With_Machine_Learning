import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import yfinance as yf
from keras.layers import Dropout
from lstm import LSTMModel
from keras.optimizers import Adam
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_stock_data(ticker):
    yf.pdr_override()
    try:
        df = yf.download(ticker, start="2010-01-01", end="2023-11-17")
        # print(df)
    except Exception as e:
        print(f'An error has occured: {e}')
    return df

def plot(train, validation, ticker):
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel(f'Close Price USD ($) for {ticker}')
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Prediction'],loc='lower right')
    plt.show() 
    return

def preprocess(dataset):
    training_data_len = math.ceil(len(dataset)*0.8)
    
    #Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    #Creating training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = [] # Indepdent variables
    y_train = [] # Dependent variables
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0]) # Past 60 values
        y_train.append(train_data[i,0]) # 61st value we want to predict (Label)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train

def main():
    ticker = 'MSFT'
    df = get_stock_data(ticker)
    data_df= df.filter(['Close'])
    dataset = data_df.values

    x_train, y_train = preprocess(dataset)
    
    lstm_predictor = LSTMModel(input_shape=(x_train.shape[1],1))    
    lstm_predictor.forward(x_train, y_train, epochs=1, batch_size=1)
    
    # Creating testing dataset
    test_data = scaled_data[training_data_len-60:, :]
    x_test = []
    y_test = dataset[training_data_len:,:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])
        
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    # predictions = model.predict(x_test)
    predictions = lstm_predictor.predict(x_test)
    
    # Unscaling value
    predictions = scaler.inverse_transform(predictions)
    
    # Get mean squared error 
    rmse = np.sqrt(np.mean((predictions-y_test)**2))
    print(f"The mean squared error is: {rmse}")
    
    # Plot the data
    train = data_df[:training_data_len]
    validation = data_df[training_data_len:].copy()
    validation['Predictions'] = predictions
    print(validation)
    plot(train=train,validation=validation,ticker=ticker)
    return
    
if __name__=="__main__":
    main()