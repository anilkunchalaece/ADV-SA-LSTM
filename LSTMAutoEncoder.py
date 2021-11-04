"""
Build RNN based Auto encoder for sequence to sequence prediction
Ref - https://github.com/lkulowski/LSTM_encoder_decoder
"""

# Author: Laura Kulowski

import numpy as np
import random
import os, errno
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

from torchModels import getResults


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 2):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers,batch_first=True)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input)

        # print(F"encoder shape is {self.hidden[0].shape} , {self.hidden[0].shape}")
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(batch_size,self.num_layers, self.hidden_size),
                torch.zeros(batch_size,self.num_layers, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 2):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        # print(encoder_hidden_states[0].shape , x_input.shape)
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)     
        
        return output

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size,num_layers):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size,num_layers=num_layers)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size,num_layers=num_layers)
    

    def train(self,config,trainData,validData,lossType):
        hiddenLayerSize = config["hiddenLayerSize"]
        numLayers = config["numLayers"]
        learningRate = config["learningRate"]
        epochs = config["epochs"]
        batchSize = config["batchSize"]

        trainX , trainY = trainData
        validX, validY = validData

        trainX = torch.from_numpy(trainX).float()
        trainY = torch.from_numpy(trainY).float()

        validX = torch.from_numpy(validX).float()
        validY = torch.from_numpy(validY).float()

        trainLoader = DataLoader(TensorDataset(trainX,trainY),batch_size=batchSize,drop_last=True) # make batches of train_data
        validLoader = DataLoader(TensorDataset(validX,validY),batch_size=batchSize,drop_last=True)

        optimizer = optim.Adam(self.parameters(), lr = learningRate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=2)

        if lossType == "MSE" :
            criterion = nn.MSELoss()
        elif lossType == "MAE" :
            criterion = torch.nn.L1Loss()
        else :
            raise("Invalud lossType ! Please check lossType")

        # used to store losses for plotting
        vlArray = []
        tlArray = []

        validLossMin = float("inf") # used to check early stopping
        badEpoch = 0

        # Train and Eval loop 
        for epoch in range(epochs) :

            self.encoder.train()
            self.decoder.train()
            
            tl = [] # used to store training loss

            # Training
            for i, data in enumerate(trainLoader) :
                
                _trainX , _trainY = data[0] , data[1]
                
                if len(_trainY.size()) == 2 :
                    _trainY = _trainY.view(_trainX.size(0),-1,_trainX.size(-1)) # convert 2D to 3D used for single time step prediction
                
                optimizer.zero_grad()

                # outputs tensor
                outputs = torch.zeros(batchSize,_trainY.size(1), _trainY.size(-1))

                # initialize hidden state
                encoder_hidden = self.encoder.init_hidden(batchSize)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(_trainX)

                # decoder with teacher forcing
                decoder_input = _trainX[:,-1, :].view(_trainX.size(0),1,-1)   # get output and reshape it to (batch_size,1,input) 1 is single timestep i.e last time step in input
                decoder_hidden = encoder_hidden

                # print(F"_trainX shape is {_trainX.shape} , _trainY shape is {_trainY.shape}")

                # predict recursively
                for t in range(_trainY.size(1)): 
                    decoder_output = self.decoder(decoder_input, decoder_hidden)
                    # print(decoder_output.shape)
                    outputs[:,t,:] = decoder_output.view(decoder_output.size(0),-1)
                    decoder_input = decoder_output

                # print(outputs.shape,_trainY.shape)
                loss = criterion(outputs,_trainY)
                loss.backward()
                optimizer.step()
                tl.append(loss.item())

            # Eval
            vl = []
            # hiddenValid = None
            self.encoder.eval() # change the model into eval mode
            self.decoder.eval() 

            for _validX , _validY in validLoader :
                if len(_validY.size()) == 2 :
                    _validY = _validY.view(_validX.size(0),-1,_validX.size(-1)) # convert 2D to 3D used for single time step prediction

                encoder_output_val, encoder_hidden_val = self.encoder(_validX)
                
                # initialize tensor for predictions
                outputs_val = torch.zeros(batchSize,_trainY.size(1), _trainY.size(-1))

                # decode input_tensor
                decoder_input_val = _validX[:,-1, :].view(_validX.size(0),-1,_validX.size(-1))
                decoder_hidden_val = encoder_hidden_val
                
                for t in range(_validY.size(1)):
                    decoder_output_val = self.decoder(decoder_input_val, decoder_hidden_val)
                    outputs_val[:,t,:] = decoder_output_val.view(batchSize,-1)
                    decoder_input_val = decoder_output_val                                
                # print(outputs_val.shape,_validY.shape)
                loss = criterion(outputs_val,_validY)
                vl.append(loss.item())

            trainLoss = np.mean(tl)
            validLoss = np.mean(vl)

            tlArray.append(trainLoss)
            vlArray.append(validLoss)

            scheduler.step(validLoss) # ReduceLROnPlateau ref - https://pytorch.org/docs/stable/optim.html


            if epoch % 30 == 0 : # print for only every 10 epochs
                print(F"epoch {epoch}/{epochs} tl,vl => {trainLoss}, {validLoss}")

            # Early stopping to avoid overfitting
            if validLoss < validLossMin :
                validLossMin = validLoss # save current validLoss
                badEpoch = 0
                
                #save the model
                torch.save(self.encoder.state_dict(),config["modelName"]+"enc.pth")
                torch.save(self.decoder.state_dict(),config["modelName"]+"dec.pth")
            else :
                if validLoss - validLossMin >= config["earlyStoppingDelta"] :
                    badEpoch += 1 # increment badEpoch

                if badEpoch >= config["patience"] :
                    print(F"Training stops in early due to overfitting in epoch {epoch}")
                    print(F"tl Array {tlArray[-4:]} vl array {vlArray[-4:]} ")
                    break # stop training
        return {
            "trainingLoss" : tlArray,
            "validLoss" : vlArray
        }    
        

    def test(self,config,testData) :
        # load test data
        testX , testY = testData

        testX = torch.from_numpy(testX).float()
        testY = torch.from_numpy(testY).float()

        testLoader = DataLoader(TensorDataset(testX,testY),batch_size=1)

        self.encoder.load_state_dict(torch.load(config["modelName"]+"enc.pth"))
        self.decoder.load_state_dict(torch.load(config["modelName"]+"dec.pth"))

        gt = testY
        pred = torch.tensor([])

        self.encoder.eval()
        self.decoder.eval()

        for _testX , _testY in testLoader :
            if len(_testY.size()) == 2 :
                _testY = _testY.view(_testX.size(0),-1,_testX.size(-1)) # convert 2D to 3D used for single time step prediction

            encoder_output_val, encoder_hidden_val = self.encoder(_testX)
            
            # initialize tensor for predictions
            outputs_val = torch.zeros(_testY.size(0),_testY.size(1), _testY.size(-1))

            # decode input_tensor
            decoder_input_val = _testX[:,-1, :].view(_testX.size(0),-1,_testX.size(-1))
            decoder_hidden_val = encoder_hidden_val
            
            for t in range(_testY.size(1)):
                decoder_output_val = self.decoder(decoder_input_val, decoder_hidden_val)
                outputs_val[:,t,:] = decoder_output_val.view(decoder_output_val.size(0),-1)
                decoder_input_val = decoder_output_val
            
            pred = torch.cat((pred,outputs_val),dim=1) 
        
        return np.array(gt) , pred.view(-1,pred.size(-1)).detach().numpy()

if __name__ == "__main__" :
    from modelData import ModelData # get training data
    config = {
        "learningRate" : 0.0001,
        "hiddenLayerSize" : 100,
        "numLayers" : 6,
        "batchSize" : 50,
        "epochs" : 1000,
        "plotLoss" : False,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "modelName" : "autoEncoder",
        "dataFile" : "pedx_out.json",
        "lookBack" : 5,
        "key" : "pose_and_trans", # for which data model should run on
        "resultDir" : "results"
    }

    config["modelName"] = "AutoEncoder.pth"
    # config["numLayers"] = 2

    mData = ModelData(config)

    trainData = mData.getTrainData()
    validData = mData.getValidData()
    testData = mData.getTestData()
    m = lstm_seq2seq(trainData[0].shape[-1],config["hiddenLayerSize"],config["numLayers"])
    m.train(config,trainData,validData,"MSE")
    gt , pred = m.test(config,testData)
    vae = getResults(gt,pred,steps=500) 
    print(vae)