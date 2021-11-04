import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
import numpy as np

LSTM_UNITS = 32
DROP_OUT_VAL = 0.2


"""
Model for SMPL pose
@params
    data - SMPL pose data with shape =>  noOfTrainingSamples , look_back_window, 72 ( no.of SMPL parameters )
"""
def poseModel(data):
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS,return_sequences=True,input_shape=(data.shape[1],data.shape[2])))
    model.add(Dropout(DROP_OUT_VAL))

    model.add(LSTM(units=LSTM_UNITS,return_sequences = True))
    model.add(Dropout(DROP_OUT_VAL))

    model.add(LSTM(units=LSTM_UNITS))

    model.add(Dense(units=data.shape[2]))

    return model

"""
Model for 3D joints
@params
    data - 3D joints data with shape => noOfTrainingSamples , look_back_window, 49 * 3 (49 joints , XYZ)
"""
def Joints3DModel(data):
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS,return_sequences=True,input_shape=(data.shape[1],data.shape[2])))
    model.add(Dropout(DROP_OUT_VAL))

    model.add(LSTM(units=LSTM_UNITS,return_sequences = True))
    model.add(Dropout(DROP_OUT_VAL))

    model.add(LSTM(units=LSTM_UNITS))

    model.add(Dense(units=(data.shape[2])))
    return model


"""
frameDifference given in Bio-LSTM paper
3D pedestrian posed predicted by computing the difference in translatioan and pose parameters from past frames 
and then applying that difference to the future frames. Please check Baseline methods section in Bio-LSTM paper
"""
def frameDifference(data) :
    pass