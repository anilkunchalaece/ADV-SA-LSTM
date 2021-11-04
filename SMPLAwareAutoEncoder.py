"""
This script is extension for lstm auto encoder, we use 3 seperate lstms 
"""

# Author: Laura Kulowski

import numpy as np
import random
import os, errno
import sys
from numpy.lib.function_base import bartlett

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

from torchModels import getResults


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, conf,num_layers = 2):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        # define LSTM layer
        # self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,batch_first=True)
        #lstm for rotation parameters
        self.lstm_r = nn.LSTM(input_size=conf["enc_stage1_lstm_r_input"],
                            hidden_size=conf["enc_stage1_lstm_r_hidden"],
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0.1) 
        
        # lstm for joints pose parameters
        self.lstm_p = nn.LSTM(input_size=conf["enc_stage1_lstm_p_input"],
                            hidden_size=conf["enc_stage1_lstm_p_hidden"],
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0.1)
        
        # lstm for translation parameters
        self.lstm_t = nn.LSTM(input_size=conf["enc_stage1_lstm_t_input"],
                            hidden_size=conf["enc_stage1_lstm_t_hidden"],
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0.1)

        self.stage1_fc1 = nn.Linear(in_features=conf["enc_stage1_fc1_in"],
                                out_features=conf["enc_stage1_fc1_out"])

        self.stage1_fc2 = nn.Linear(in_features=conf["enc_stage1_fc2_in"],
                                out_features=conf["enc_stage1_fc2_out"])

        self.stage2_lstm1 = nn.LSTM(input_size=conf["enc_stage2_lstm1_input"],
                                    hidden_size=conf["enc_stage2_lstm1_hidden"],
                                    num_layers=self.num_layers,
                                    dropout=0.1,
                                    batch_first=True)

        self.stage2_lstm2 = nn.LSTM(input_size=conf["enc_stage2_lstm2_input"],
                                    hidden_size=conf["enc_stage2_lstm2_hidden"],
                                    num_layers=self.num_layers,
                                    dropout=0.1,
                                    batch_first=True)
        
        self.stage2_fc1 = nn.Linear(in_features=conf["enc_stage2_fc1_in"],
                                    out_features=conf["enc_stage2_fc1_out"])

        self.stage3_lstm = nn.LSTM(input_size=conf["enc_stage3_lstm_input"],
                                    hidden_size=conf["enc_stage3_lstm_hidden"],
                                    num_layers=self.num_layers,
                                    batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyRelu = nn.LeakyReLU()


    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        # lstm_out, self.hidden = self.lstm(x_input)
        r_input = x_input[:,:,:3] # first three inputs are global rotation
        t_input = x_input[:,:,-4:] # last four elemets are translation parameters
        p_input = x_input[:,:,3:-4] # all except first 3 and last 4 are pose parameters

        # print(F"r_input -> {r_input.shape} t_input -> {t_input.shape} p_input -> {p_input.shape}")

        # we will be forwading the weights i.e input parameters representations to next stage instead of output
        lstm_r_out, (lstm_r_weights, lstm_r_cell_state) = self.lstm_r(r_input) # we are not intrested in cell state or lstm out
        lstm_p_out, (lstm_p_weights, lstm_p_cell_state) = self.lstm_p(p_input) 
        lstm_t_out, (lstm_t_weights, lstm_t_cell_state) = self.lstm_t(t_input)

        # by default weight has dimension (num_layers , batch_size, hidden_size) 
        # we need to convert this is batch_first => (batch_size,num_layers,hidden_size)
        # why not simply use view ? we can't cause its a shit ref - https://medium.com/mcgill-artificial-intelligence-review/the-dangers-of-reshaping-and-other-fun-mistakes-ive-learnt-from-pytorch-b6a5bdc1c275
        # print(lstm_r_weights.shape)
        lstm_r_weights = lstm_r_weights.transpose(0,1).contiguous()
        lstm_p_weights = lstm_p_weights.transpose(0,1).contiguous()
        lstm_t_weights = lstm_t_weights.transpose(0,1).contiguous()

        # print(F"batch first shapes \n r -> {lstm_r_weights.shape} t -> {lstm_t_weights.shape} p -> {lstm_p_weights.shape}")
        # combine r and p -> stage1_fc1_input
        # combine t and p -> stage1_fc2_input
        stage1_fc1_input = torch.cat((lstm_p_weights,lstm_r_weights),dim=2)
        stage1_fc2_input = torch.cat((lstm_p_weights,lstm_t_weights),dim=2)

        stage1_fc1_out = self.tanh(self.dropout(self.stage1_fc1(stage1_fc1_input)))
        stage1_fc2_out = self.tanh(self.dropout(self.stage1_fc2(stage1_fc2_input)))

        stage2_lstm1_out, (stage2_lstm1_weights, stage2_lstm1_cell_state) = self.stage2_lstm1(stage1_fc1_out)
        stage2_lstm2_out, (stage2_lstm2_weights, stage2_lstm2_cell_state) = self.stage2_lstm2(stage1_fc2_out)

        # combine stage2 lstm weights as single vector to feed to the stage3 fc
        # modify weights into batch_first
        stage2_lstm1_weights = stage2_lstm1_weights.transpose(0,1).contiguous()
        stage2_lstm2_weights = stage2_lstm2_weights.transpose(0,1).contiguous()

        stage2_fc1_input = torch.cat((stage2_lstm1_weights,stage2_lstm2_weights),dim=2)

        stage2_fc1_out = self.tanh(self.dropout(self.stage2_fc1(stage2_fc1_input)))

        # send stage2 fc_out to lstm
        stage3_lstm_out , stage3_lstm_hidden = self.stage3_lstm(stage2_fc1_out)
        
        return stage3_lstm_out, stage3_lstm_hidden     
    
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
    
    def __init__(self, input_size, conf,num_layers = 2):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        # self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = num_layers,batch_first=True)
        # self.linear = nn.Linear(hidden_size, input_size)
        self.lstm = nn.LSTM(input_size=conf["dec_lstm_input"],hidden_size=conf["dec_lstm_hidden"],
                            num_layers=self.num_layers,batch_first=True)
        
        self.stage1_fc1 = nn.Linear(in_features=conf["dec_stage1_fc_in"],out_features=conf["dec_stage1_fc_out"])
        # self.stage2_fc_r = nn.Linear(in_features=conf["dec_stage2_fc_r_in"],out_features=conf["dec_stage2_fc_r_out"])
        # self.stage2_fc_p = nn.Linear(in_features=conf["dec_stage2_fc_p_in"],out_features=conf["dec_stage2_fc_p_out"])
        # self.stage2_fc_t = nn.Linear(in_features=conf["dec_stage2_fc_t_in"],out_features=conf["dec_stage2_fc_t_out"])
        # self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()


    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        # print(encoder_hidden_states[0].shape , x_input.shape)
        # lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        # output = self.linear(lstm_out)
        lstm_out, (lstm_weights, lstm_cell_state) = self.lstm(x_input,encoder_hidden_states)
        # lstm_weights = lstm_weights.transpose(0,1).contiguous()

        stage1_fc1_out = self.stage1_fc1(lstm_out)

        # stage2_fc_r_out = self.stage2_fc_r(stage1_fc1_out)
        # stage2_fc_p_out = self.stage2_fc_p(stage1_fc1_out)
        # stage2_fc_t_out = self.stage2_fc_t(stage1_fc1_out)
        # # print(F"stage2_fc_r_out -> {stage2_fc_r_out.size()} , stage2_fc_p_out -> {stage2_fc_p_out.size()} , stage2_fc_t_out -> {stage2_fc_t_out.size()}")
        # output = torch.cat((stage2_fc_r_out,stage2_fc_p_out,stage2_fc_t_out),dim=2)
        # # print(output.shape)
        # output = torch.squeeze(stage1_fc1_out,dim=1)
        return stage1_fc1_out

class SMPLAwareAE(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size,num_layers):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(SMPLAwareAE, self).__init__()

        self.input_size = input_size
        conf = self.getExtraParameters()

        self.encoder = lstm_encoder(input_size=input_size,conf=conf,num_layers=num_layers)
        self.decoder = lstm_decoder(input_size=input_size,conf=conf,num_layers=num_layers)
    
    def getExtraParameters(self):
        return {
            "enc_stage1_lstm_r_hidden" : 2,
            "enc_stage1_lstm_r_input" : 3,
            "enc_stage1_lstm_t_hidden" : 2,
            "enc_stage1_lstm_t_input" : 4,
            "enc_stage1_lstm_p_hidden" : 10,
            "enc_stage1_lstm_p_input" : 69,
            "enc_stage1_fc1_in" : 12,
            "enc_stage1_fc2_in" : 12,
            "enc_stage1_fc1_out" : 50,
            "enc_stage1_fc2_out" : 50,
            "enc_stage2_lstm1_input" : 50,
            "enc_stage2_lstm2_input" : 50,
            "enc_stage2_lstm1_hidden" : 10,
            "enc_stage2_lstm2_hidden" : 10,
            "enc_stage2_fc1_in": 20,
            "enc_stage2_fc1_out": 16,
            "enc_stage3_lstm_input" : 16,
            "enc_stage3_lstm_hidden": 8,
            "dec_lstm_hidden":8,
            "dec_lstm_input" : 76,
            "dec_stage1_fc_in" : 8,
            "dec_stage1_fc_out" : 76
        }
    

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
            raise("Invalid loss type ! Please check loss type")

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
                
                if len(_trainY.size()) == 2 : # if trainY has only 2 dim i.e Batchsize,SampleSize , we are only predicting single step
                    _trainY = _trainY.view(_trainX.size(0),-1,_trainX.size(-1)) # convert 2D to 3D used for single time step prediction
                
                optimizer.zero_grad()

                # outputs tensor
                outputs = torch.zeros(batchSize,_trainY.size(1), _trainY.size(-1))

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batchSize)

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
                # using weighted loss
                # loss = 0.2*criterion(outputs[:,:,:3],_trainY[:,:,:3]) + 0.6*criterion(outputs[:,:,3:-4],_trainY[:,:,3:-4]) + 0.2*criterion(outputs[:,:,-4:],_trainY[:,:,-4:])
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
                # loss = 0.2*criterion(outputs_val[:,:,:3],_validY[:,:,:3]) + 0.6*criterion(outputs_val[:,:,3:-4],_validY[:,:,3:-4]) + 0.2*criterion(outputs_val[:,:,-4:],_validY[:,:,-4:])
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
                torch.save(self.encoder.state_dict(),config["modelName"].replace(".pth","")+"_enc.pth")
                torch.save(self.decoder.state_dict(),config["modelName"].replace(".pth","")+"_dec.pth")
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

        self.encoder.load_state_dict(torch.load(config["modelName"]+"_enc.pth"))
        self.decoder.load_state_dict(torch.load(config["modelName"]+"_dec.pth"))

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
            decoder_input_val = _testX[:,-1, :].view(_testX.size(0),-1,_testX.size(-1))#the value used as input to decoder i.e last time step
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
        "numLayers" : 2,
        "batchSize" : 50,
        "epochs" : 1000,
        "plotLoss" : False,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "modelName" : "autoEncoder",
        "dataFile" : "modified_pedx_out.json",
        "lookBack" : 5,
        "key" : "pose_and_orig_cam", # for which data model should run on
        "resultDir" : "results"
    }

    lossType = "MSE"

    config["modelName"] = "SMPLAwareAutoEncoder.pth"
    # config["numLayers"] = 2

    mData = ModelData(config)

    trainData = mData.getTrainData()
    validData = mData.getValidData()
    testData = mData.getTestData()
    m = SMPLAwareAE(trainData[0].shape[-1],config["numLayers"])
    m.train(config,trainData,validData,lossType)
    gt , pred = m.test(config,testData)
    shapeV = ((np.random.rand(1,10) - 0.5)*0.06) # random shape value used for all models for testing
    vae = getResults(gt,pred,shapeV,steps=200) 
    print(vae)