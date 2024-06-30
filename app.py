# pip install pandas_datareader
# pip install --upgrade holidays
# pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
# pip install seaborn
# pip install tensorflow cls

# pip install keras
# pip install streamlit    
# pip install pandas_ta
# ! pip install plotly

import pandas_ta as ta    
import pandas as pd
import numpy as np
import math
import datetime as dt
import seaborn as sns
import os.path
import holidays
import streamlit as st

from tvDatafeed import TvDatafeed, Interval

from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import keras
from tf_keras.optimizers.legacy import Adam
from tf_keras.models import Sequential
from tf_keras.layers import Dense, GRU, LSTM
# from keras._tf_keras.keras.optimizers import Adam
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import Dense, GRU, LSTM

from itertools import cycle

import plotly.express as px

# Get New stock data
def getNewData(stock_name):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=stock_name.upper(), exchange='MYX', interval=Interval.in_daily, n_bars=2000)
    if data is not None:
        del data['symbol']
        data = calculate_indicator(data)
        # data = data[:len(data)-30]
        # Save processed stock data into a csv
        data.to_csv('stockprocesseddata/' + stock_name + '_processed.csv')
    return data

# Get Processed stock data
def getProcessedData(stock_name):
    fname = 'stockprocesseddata/' + stock_name + '_processed.csv'
    if (os.path.isfile(fname)):
        data = pd.read_csv('stockprocesseddata/' + stock_name + '_processed.csv')
        data['datetime'] = pd.to_datetime(data["datetime"]).dt.normalize()
        data.set_index('datetime', inplace=True)
        return data


# Convert date from YYYYMMDD to DD-MM-YYYY
def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')


def rsi(values):
    up = values[values > 0].mean()
    down = -1 * values[values < 0].mean()
    return 100 * up / (up + down)


def calculate_indicator(data):
    # Get multiple technical indicator
    ma_day = [10, 50, 100]

    for ma in ma_day:
        column_name = "MA for %s days" % (str(ma))
        data[column_name] = pd.DataFrame.rolling(data['close'], ma).mean()

    data['26_ema'] = data['close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=False).mean()
    data['12_ema'] = data['close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=False).mean()
    data['MACD'] = data['12_ema'] - data['26_ema']
    data['Momentum_1D'] = (data['close'] - data['close'].shift(1)).fillna(0)
    data['RSI_14D'] = data['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)

    data = data.fillna(0)
    del data['Momentum_1D']
    return data


def prep_training(dataset, look_back):
    # Prepare the training data
    x_train = []
    y_train = []

    for i in range(look_back, dataset.shape[0]):
        x_train.append(dataset[i - look_back:i, :])
        y_train.append(dataset[i, :4])

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


def prep_testing(dataset, look_back):
    # Prepare the test data
    x_test = []
    y_test = []

    for i in range(look_back, dataset.shape[0]):
        x_test.append(dataset[i - look_back:i, :])
        y_test.append(dataset[i, :4])
    x_test, y_test = np.array(x_test), np.array(y_test)
    return x_test, y_test


def train_model(stock_name, time_step):

    st.subheader('Model Training')

    st.write('Training Model.......Please Wait For A while')
    st.write('It may takes a few minutes.')

    look_back = time_step

    # get dataset
    stock_data = getProcessedData(stock_name)
    stock = getProcessedData(stock_name)
    resultdf = getProcessedData(stock_name)

    if ('date') in stock:
        del stock['date']
    if ('Momentum_1D') in stock:
        del stock['Momentum_1D']
    del stock['volume']

    # normalize the dataset
    # input
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock = scaler.fit_transform(stock)

    # output
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    resultdf = resultdf[['close', 'open', 'high', 'low']]
    result = scaler2.fit_transform(np.array(resultdf).reshape(-1, 1))

    # split into train and test sets
    train_size = int(len(stock) * 0.8)
    test_size = len(stock) - train_size
    train, test = stock[0:train_size, :], stock[train_size:len(stock), :]

    look_back = time_step
    trainX, trainY = prep_training(train, look_back)
    testX, testY = prep_testing(test, look_back)

    features = stock.shape[1]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, features))
    testX = np.reshape(testX, (testX.shape[0], look_back, features))

    tf.keras.backend.clear_session()
    # keras.utils.clear_session(free_memory=True)
    # 64 unit
    # return sequence true = return all hidden state to pass to next
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    opt = Adam(learning_rate=0.001, decay=1e-6)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(trainX.shape[1:])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(16, 'relu'))
    model.add(Dense(4, 'relu'))
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()

    predict_model = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=25, batch_size=15)
    model.save("stockpredictmodel/" + stock_name + ".h5")

    print('\n# Evaluate on test data')
    results = model.evaluate(testX, testY, batch_size=30)
    print('test loss, test acc:', results)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    # Transform back to original form
    trainPredict = scaler2.inverse_transform(trainPredict)
    testPredict = scaler2.inverse_transform(testPredict)
    trainY = scaler2.inverse_transform(trainY)
    testY = scaler2.inverse_transform(testY)
    original_ytrain = scaler2.inverse_transform(trainY.reshape(-1, 1))
    original_ytest = scaler2.inverse_transform(testY.reshape(-1, 1))

    trainPredictPlot = np.empty_like(stock)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :4] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(stock)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2):len(stock), :4] = testPredict

    train_array = trainPredictPlot[:, :1].reshape(-1, 1)
    train_frame = pd.DataFrame(train_array, columns=['train_predicted_close'])

    test_array = testPredictPlot[:, :1].reshape(-1, 1).tolist()
    test_frame = pd.DataFrame(test_array, columns=['test_predicted_close'])

    st.subheader('Training VS Testing')

    names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])
    plotdf = pd.DataFrame({'date': stock_data.index,
                           'original_close': stock_data['close'].values,
                           'train_predicted_close': train_frame['train_predicted_close'].values,
                           'test_predicted_close': test_frame['test_predicted_close'].values})

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                               plotdf['test_predicted_close']],
                  labels={'value': 'Stock price', 'date': 'Date'}, color_discrete_sequence=['blue', 'cyan', 'red'])

    fig.update_layout(
        title_text='Comparision between original close price vs predicted close price of ' + stock_name.upper(),
        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #fig.show()
    st.subheader('LSTM Network : ')
    st.plotly_chart(fig)

def process_data(data, time_step):
    stock = data
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock = scaler.fit_transform(stock)

    dataset = prep_data(stock, time_step)
    features = data.shape[1]
    source = np.reshape(dataset, (dataset.shape[0], time_step, features))

    return source


def getnextweekday(date):
    nextdate = date + dt.timedelta(days=1)
    weekday = nextdate[0].weekday()

    while weekday > 4 or nextdate in holidays.MY(years=nextdate.year).items():
        nextdate = nextdate + dt.timedelta(days=1)
        weekday = nextdate[0].weekday()

    return nextdate


def prediction(stock_name, data, time_step, duration):
    print('Start predicting the next ' + str(duration) + ' days....')

    temp_stock = getProcessedData(stock_name)
    if ('Momentum_1D') in temp_stock:
        del temp_stock['Momentum_1D']
    del temp_stock['volume']

    datetime = temp_stock.tail(1).index

    resultdf = getProcessedData(stock_name)
    scaler = MinMaxScaler(feature_range=(0, 1))
    resultdf = resultdf[['close', 'open', 'high', 'low']]
    result = scaler.fit_transform(np.array(resultdf).reshape(-1, 1))

    i = 0
    while i < duration:
        if i == 0:
            print('Day 1 : ')
            x_input = data[len(data) - time_step:]
            model = tf.keras.models.load_model("stockpredictmodel/" + stock_name + ".h5")
            # model = keras.layers.TFSMLayer("stockpredictmodel/" + stock_name + ".h5")
            yhat = model.predict(x_input)
            yhat = scaler.inverse_transform(yhat)
            new_close = yhat[0, 0]
            new_open = yhat[0, 1]
            new_high = yhat[0, 3]
            new_low = yhat[0, 2]
            datetime = getnextweekday(datetime)
            df = {'datetime': datetime, 'close': [new_close], 'open': [new_open], 'high': [new_high], 'low': [new_low]}
            df = pd.DataFrame(df)
            df.set_index('datetime', inplace=True)
            temp_stock = pd.concat([temp_stock, df])
            i += 1
        else:
            print('Day ' + str(i + 1) + ' : ')
            data = calculate_indicator(temp_stock)
            data = process_data(data, time_step)
            x_input = data[len(data) - time_step:]
            model = tf.keras.models.load_model("stockpredictmodel/" + stock_name + ".h5")
            # model = keras.layers.TFSMLayer("stockpredictmodel/" + stock_name + ".h5")
            yhat = model.predict(x_input)
            yhat = scaler.inverse_transform(yhat)
            new_close = yhat[0, 0]
            new_open = yhat[0, 1]
            new_high = yhat[0, 3]
            new_low = yhat[0, 2]
            datetime = getnextweekday(datetime)
            df = {'datetime': datetime, 'close': [new_close], 'open': [new_open], 'high': [new_high], 'low': [new_low]}
            df = pd.DataFrame(df)
            df.set_index('datetime', inplace=True)
            temp_stock = pd.concat([temp_stock, df])
            i += 1

    return temp_stock


def prep_data(dataset, look_back):
    # Prepare the training data
    x_train = []

    for i in range(look_back, dataset.shape[0]):
        x_train.append(dataset[i - look_back:i, :])

    x_train = np.array(x_train)
    return x_train


def data_to_predict(stock_name, time_step):
    data = getProcessedData(stock_name)
    stock = getProcessedData(stock_name)
    resultdf = getProcessedData(stock_name)

    if ('Momentum_1D') in data:
        del data['Momentum_1D']
    del data['volume']

    if ('Momentum_1D') in stock:
        del stock['Momentum_1D']
    del stock['volume']

    # normalize the dataset
    # input
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock = scaler.fit_transform(stock)
    # output
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    resultdf = resultdf[['close', 'open', 'high', 'low']]
    result = scaler2.fit_transform(np.array(resultdf).reshape(-1, 1))

    dataset = prep_data(stock, time_step)
    features = data.shape[1]
    source = np.reshape(dataset, (dataset.shape[0], time_step, features))

    return source


def plot_graph(stock_name, output, time_step, duration):
    output = output.close
    temp_stock = getProcessedData(stock_name)
    ori_data = output[len(temp_stock) - (2 + time_step):len(temp_stock)]
    predicted_data = output[len(temp_stock) - 1:]

    portion = output[len(temp_stock) - (2 + time_step):]
    print(portion.shape)
    originalPlot = np.empty_like(portion)
    originalPlot[:] = np.nan
    originalPlot[:len(ori_data)] = ori_data

    # shift test predictions for plotting

    predictedPlot = np.empty_like(portion)
    predictedPlot[:] = np.nan
    predictedPlot[len(ori_data) - 1:] = predicted_data

    original_array = originalPlot[:].reshape(-1, 1).tolist()
    original_frame = pd.DataFrame(original_array, columns=['original_close'])

    predicted_array = predictedPlot[:].reshape(-1, 1).tolist()
    predicted_frame = pd.DataFrame(predicted_array, columns=['predicted_close'])

    names = cycle(['Original Close', 'Predicted Close'])
    plotdf = pd.DataFrame({
        'date': portion.index,
        'original_close': original_frame['original_close'].values,
        'predicted_close': predicted_frame['predicted_close'].values
    })

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['predicted_close']],
                  labels={'value': 'Stock price', 'date': 'Date'}, color_discrete_sequence=['red', 'blue'])

    fig.update_layout(title_text='Predicted close price of ' + stock_name.upper(),
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #fig.show()
    st.subheader('Prediction Result : ')

    st.subheader('LSTM Network : ')
    st.plotly_chart(fig)

def start_prediction(stock_name, time_step, duration):
    time_step = int(time_step)
    duration = int(duration)

    
    data = getNewData(stock_name)
    if data is not None:
        train_model(stock_name, time_step)
        source = data_to_predict(stock_name, time_step)
        close_output = prediction(stock_name, source, time_step, duration)
        plot_graph(stock_name, close_output, time_step, duration)
    else: 
       st.warning('Stock Not Exist. Please Try Again')


#######################################################################################################
#GRU MODEL
#Get Stock Data
def gru_stock_data(stock_name):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=stock_name.upper(),exchange='MYX',interval=Interval.in_daily,n_bars=2000)
    if data is not None:
        del data['symbol']
        data = gru_indicators(data)
        # data = data[:len(data)-30]    
        # Save processed stock data into a csv
        data.to_csv('stockprocesseddata/gru_'+ stock_name +'_processed.csv')
    return data

# Get Processed stock data
def gru_processed_data(stock_name):
    fname ='stockprocesseddata/gru_' + stock_name + '_processed.csv'; 
    if(os.path.isfile(fname)):
        data = pd.read_csv('stockprocesseddata/gru_' + stock_name + '_processed.csv')
        data['datetime'] = pd.to_datetime(data["datetime"]).dt.normalize()
        data.set_index('datetime' ,inplace=True)
        return data
    
#Add indicators using pandas_ta library 
def gru_indicators(data):
    data['RSI'] = data.ta.rsi()
    data.ta.macd(append = True)
    data.ta.stoch(append = True)
    data.ta.sma(append = True)
    data.ta.ema(append = True)
    data = data.fillna(0)    
    return data

def gru_process_data(data, predict_days):
    time_step = predict_days
    stock = data
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock = scaler.fit_transform(stock)
        
    dataset = prep_data(stock, time_step)
    features = data.shape[1]
    transformed = np.reshape(dataset, (dataset.shape[0], time_step, features))
    
    return transformed

def gru_get_x_data(stock_name, time_step):
    data = gru_processed_data(stock_name)
    stock = gru_processed_data(stock_name)
     
    if ('volume') in data:
        del data['volume']   
    
    if ('volume') in stock:
        del stock['volume']
    
    # normalize the dataset
    #input 
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock = scaler.fit_transform(stock)
       
    dataset = prep_data(stock, time_step)
    features = data.shape[1]
    x_data = np.reshape(dataset, (dataset.shape[0], time_step, features))
    
    return x_data

def gru_model_training(stock_name,predict_days):
    
    look_back = predict_days
    
    # Dataset
    stock_data = gru_processed_data(stock_name)
    model_data = gru_processed_data(stock_name)
    output_data = gru_processed_data(stock_name)
    
    # normalize the dataset    
    if ('date') in model_data:
        del model_data['date']
    if('volume') in model_data:
        del model_data['volume']        
        
    # Perform Normalization    
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_data = scaler.fit_transform(model_data)
    
    scaler2 = MinMaxScaler(feature_range=(0, 1))    
    output_data = output_data[['close','open','high','low']]
    output = scaler2.fit_transform(np.array(output_data).reshape(-1,1))
    
    #split into train and test sets
    train_size = int(len(model_data) * 0.8) 
    test_size = len(model_data) - train_size
    train, test = model_data[0:train_size, :], model_data[train_size:len(model_data), :]

    trainX, trainY = prep_training(train, look_back) 
    testX, testY = prep_testing(test, look_back)
    
    # reshape input to be  [samples, time steps, features]
    features = model_data.shape[1]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, features))
    testX = np.reshape(testX, (testX.shape[0],look_back, features))
    
    tf.keras.backend.clear_session()
    # keras.utils.clear_session(free_memory=True)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    opt = Adam(learning_rate=0.001, decay=1e-6)
    # The GRU architecture
    gru_model = Sequential()
    # First GRU layer 
    gru_model.add(GRU(units=64, return_sequences=True, input_shape=(trainX.shape[1:]), activation='tanh'))    
    # Second GRU layer
    gru_model.add(GRU(units=64, return_sequences=True, activation='tanh'))    
    # Third GRU layer
    gru_model.add(GRU(units=64, return_sequences=True, activation='tanh'))
    # Fourth GRU layer
    gru_model.add(GRU(units=64, activation='tanh'))    
    # Dense layer
    gru_model.add(Dense(units=4,))
    # Compiling the GRU
    gru_model.compile(loss='mean_squared_error',optimizer=opt)
    #gru_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    gru_model.summary()
    
    predict_model = gru_model.fit(trainX, trainY, validation_data=(testX,testY), epochs=25, batch_size=15)
    
    # Save the model into a h5 file
    gru_model.save("stockpredictmodel/gru_trainned_" + stock_name + ".h5")
    
    # calculate root mean squared error
    trainPredict = gru_model.predict(trainX)
    testPredict = gru_model.predict(testX)
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # Transform back to original form
    trainPredict = scaler2.inverse_transform(trainPredict)
    testPredict = scaler2.inverse_transform(testPredict)
    trainY = scaler2.inverse_transform(trainY)
    testY = scaler2.inverse_transform(testY)
    original_ytrain = scaler2.inverse_transform(trainY.reshape(-1,1)) 
    original_ytest = scaler2.inverse_transform(testY.reshape(-1,1)) 
    
    trainPredictPlot = np.empty_like(model_data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :4] = trainPredict

    print("Train predicted data: ", trainPredictPlot.shape)


    # shift test predictions for plotting
    testPredictPlot = np.empty_like(model_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(model_data), :4] = testPredict
    print("Test predicted data: ", testPredictPlot.shape)
    
    train_array = trainPredictPlot[:,:1].reshape(-1,1)
    train_frame = pd.DataFrame(train_array, columns=['train_predicted_close'])

    test_array = testPredictPlot[:,:1].reshape(-1,1).tolist()
    test_frame = pd.DataFrame(test_array, columns=['test_predicted_close'])

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
    plotdf = pd.DataFrame({'date': stock_data.index,
                           'original_close': stock_data['close'].values,
                           'train_predicted_close': train_frame['train_predicted_close'],
                          'test_predicted_close': test_frame['test_predicted_close']})

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                                  plotdf['test_predicted_close']],
                      labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                          plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #fig.show()
    st.subheader('GRU Network : ')
    st.plotly_chart(fig)
    
#gru future prediction    
def gru_future_prediction(stock_name, data, look_back, predict_days):
    stock = gru_processed_data(stock_name)
    
    if ('volume') in stock:
        del stock['volume'] 
        
    datetime = stock.tail(1).index
    print(datetime)  
    
    output_data = gru_processed_data(stock_name)
    scaler = MinMaxScaler(feature_range=(0, 1))
    output_data = output_data[['close','open','high','low']]
    output = scaler.fit_transform(np.array(output_data).reshape(-1,1))
     
    i = 0
    while i < predict_days:
        if i == 0:               
            x_input = data[len(data)-look_back:]
            # model = tf.keras.models.load_model("stockpredictmodel/gru_trainned_" + stock_name + ".h5")
            model = keras.layers.TFSMLayer("stockpredictmodel/" + stock_name + ".h5")
            future = model.predict(x_input) 
            future = scaler.inverse_transform(future)
            next_close = future[0,0]
            next_open = future[0,1]
            next_high = future[0,3] 
            next_low = future[0,2] 
            datetime = getnextweekday(datetime)
            df = {'datetime': datetime, 'close': [next_close],'open':[next_open],'high':[next_high], 'low':[next_low]}
            df = pd.DataFrame(df)    
            df.set_index('datetime' ,inplace=True)
            stock = pd.concat([stock,df], ignore_index=True)
            i += 1
        else:
            print('Day ' + str(i+1) + ' : ')            
            data = gru_indicators(stock)  
            data = gru_process_data(data, look_back)   
            x_input = data[len(data)-look_back:]
            # model = tf.keras.models.load_model("stockpredictmodel/gru_trainned_" + stock_name + ".h5")
            model = keras.layers.TFSMLayer("stockpredictmodel/" + stock_name + ".h5")
            future = model.predict(x_input) 
            future = scaler.inverse_transform(future)
            next_close = future[0,0]
            next_open = future[0,1]
            next_high = future[0,3]
            next_low = future[0,2] 
            df = {'datetime': datetime, 'close': [next_close],'open':[next_open],'high':[next_high], 'low':[next_low]}
            df = pd.DataFrame(df)    
            df.set_index('datetime' ,inplace=True)
            stock = pd.concat([stock,df], ignore_index=True)
            i += 1            
    return stock

def gru_plot_prediction(stock_name, output, time_step, duration):    
    output = output.close
    temp_stock = gru_processed_data(stock_name)
    ori_data = output[len(temp_stock) - ( 2 + time_step ):len(temp_stock)]
    predicted_data = output[len(temp_stock)-1:]

    portion = output[len(temp_stock) - ( 2 + time_step ):]
    print(portion.shape)
    originalPlot = np.empty_like(portion)
    
    originalPlot[:] = np.nan    
    
    originalPlot[:len(ori_data)] = ori_data  
       
    predictedPlot = np.empty_like(portion)
        
    predictedPlot[:] = np.nan
        
    predictedPlot[len(ori_data)-1:] = predicted_data
    
    original_array = originalPlot[:].reshape(-1,1).tolist()
    original_frame = pd.DataFrame(original_array, columns=['original_close'])

    
    predicted_array = predictedPlot[:].reshape(-1,1).tolist()
    predicted_frame = pd.DataFrame(predicted_array, columns=['predicted_close'])
    
    names = cycle(['Original Close','Predicted Close'])
    plotdf = pd.DataFrame({
                           'date': portion.index,
                           'original_close': original_frame['original_close'].values,
                           'predicted_close': predicted_frame['predicted_close'].values
                            })
    
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'], plotdf['predicted_close']],
                      labels={'value':'Stock price','date': 'Date'})

    
    fig.update_layout(title_text='Predicted close price of ' + stock_name.upper(),
                          plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    #fig.show()
    st.subheader('Prediction Result : ')
    st.subheader('GRU Network : ')
    st.plotly_chart(fig)

def start_gru_prediction(stock_name, time_step, duration):
    time_step = int(time_step)
    duration = int(duration)
    gru_data = gru_stock_data(stock_name)
    if gru_data is not None:
        gru_model_training(stock_name, time_step)
        x_data = gru_get_x_data(stock_name , time_step)
        result = gru_future_prediction(stock_name, x_data, time_step, duration)
        gru_plot_prediction(stock_name, result, time_step, duration)
    else: 
       st.warning('Stock Not Exist. Please Try Again')

def main():
    from datetime import datetime

    st.title('Stock Trend Prediction')

    stock_name = st.text_input("Enter a malaysia stock symbol to predict : ")
    time_step = st.text_input("Enter the number of days to be used to predict the next day  : ")
    if not time_step.isnumeric():
        st.warning('Please enter numeric character')
    
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    duration = st.text_input("Enter the number of days to predict from " + date + " : ")
    if not duration.isnumeric():
        st.warning('Please enter numeric character.')
    
    if duration.isnumeric() and time_step.isnumeric():
        if st.button('Start Predict using LSTM Model'):
            start_prediction(stock_name,time_step,duration)
        elif st.button('Start Predict using GRU Model'):
            start_gru_prediction(stock_name,time_step,duration)
    return

main()

