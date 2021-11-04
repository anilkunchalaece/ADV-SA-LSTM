"""
Author : Kunchala Anil
Date : 16 Dec 2020
"""

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import argparse

import preprocess
from modelsBioLSTM import *


def main(args):
    look_back = args.look_back
    data = preprocess.loadData(args.data_file,look_back)['joints_3d'] # lookback will be +1 since last one will be output
    print(F"total number of samples are {data.shape[0]}, with each sample containing shape of {data.shape[1]}x{data.shape[2]} i.e we have {data.shape[1]} number of timestamps data with each having length of {data.shape[2]}")
    # data = np.array(data)
    # totalSamples , look_back , data_size = data.shape
    # reshaping so we can scale the data
    # print(F"data shape is {data.shape} we are reshaping so we can scale the data")
    # data = np.reshape(data,(totalSamples*look_back,data_size))
    # print(F"data is reshaped to {data.shape}")
    # scaler = MinMaxScaler(feature_range=(0,1))
    # data_scaled = scaler.fit_transform(data)
    
    # print(F"scaled data having shape {data_scaled.shape}")

    # data_scaled = np.reshape(data,(totalSamples,look_back,data_size))

    train_data , test_data = train_test_split(data,shuffle=True)

    print(train_data.shape,test_data.shape)

    x_train = train_data[:,:5,:] # take first 4
    y_train = train_data[:,-1,:] # and last one as output
    print(F"x_train shape {x_train.shape} y_train shape {y_train.shape}")

    # model = Sequential()
    # model.add(LSTM(units=32, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units=32,return_sequences = True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units=32))


    # model.add(Dense(units=x_train.shape[2]))

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=False)

    model = Joints3DModel(x_train)

    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x_train,y_train,epochs=100,batch_size=50,validation_split=0.2,callbacks=[tensorboard],shuffle=True)   


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--look_back',type=int,default=5,help="lookback window length , l ")
    parser.add_argument('--data_file',type=str,default="out.json",help="json file which contains SMPL data")
    args = parser.parse_args()
    main(args)



