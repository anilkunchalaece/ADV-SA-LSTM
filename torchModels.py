from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import preprocess
import numpy as np
import matplotlib.pyplot as plt
from modelData import ModelData # get training data
import utils.smpl_torch_batch as smpl_torch
import evalModel
import os,json
from timeDistributed import TimeDistributed
import datetime

"""

2LR-LSTM layer - Two stacked LSTM with Fully Connected Layer

"""
class BasicLSTM(nn.Module):
    
    def __init__(self,inputSize,hiddenLayerSize,outputSize,numLayers,model):
        super(BasicLSTM,self).__init__()
        self.model = model
        self.hidden_size = hiddenLayerSize
        if  self.model == "2lr" :
            self.lstm1 = nn.LSTM(input_size=inputSize,hidden_size=hiddenLayerSize, num_layers=numLayers,batch_first=True,dropout=0.1) # noof recurrent layers, num_layers is used to created stacked lstm. 
            self.fc1 = nn.Linear(in_features=hiddenLayerSize,out_features=outputSize)
        
        elif self.model == "smpl_aware" :
            self.lstm_r = nn.LSTM(input_size=3,hidden_size=2,num_layers=numLayers,batch_first=True)
            self.lstm_p = nn.LSTM(input_size=69,hidden_size=8,num_layers=numLayers,batch_first=True)
            self.lstm_t = nn.LSTM(input_size=4,hidden_size=2,num_layers=numLayers,batch_first=True)
            # self.fc_r = nn.Linear(in_features=2,out_features=3)
            # self.fc_p = nn.Linear(in_features=8,out_features=69)
            # self.fc_t = nn.Linear(in_features=2,out_features=4)
            self.fc = nn.Linear(in_features=12,out_features=76)

        elif self.model == "ann_2lr":
            self.fc1 = nn.Linear(in_features=inputSize,out_features=500)
            self.fc2 = nn.Linear(in_features=500,out_features=1024)
            self.fc3 = nn.Linear(in_features=1024,out_features=2048)
            self.maxPool = nn.MaxPool1d(kernel_size=3)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout()
            self.lstm1 = nn.LSTM(input_size=682,hidden_size=hiddenLayerSize, num_layers=numLayers,batch_first=True,dropout=0.1) # noof recurrent layers, num_layers is used to created stacked lstm
            self.fc4 = nn.Linear(in_features=hiddenLayerSize,out_features=outputSize)
        
        elif self.model == "ann" :
            self.fc1 = nn.Linear(in_features=inputSize*5,out_features=500)# 5 is lookback - TODO- need to change it to dynamic
            self.fc2 = nn.Linear(in_features=500,out_features=1024)
            self.fc3 = nn.Linear(in_features=1024,out_features=500)
            self.fc4 = nn.Linear(in_features=500,out_features=outputSize)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p=0.5)

        elif self.model == "cnn_2lr" :
            self.cnn1 = nn.Conv1d(in_channels=1,out_channels=20,kernel_size=1)
            self.cnn2 = nn.Conv1d(in_channels=20,out_channels=40,kernel_size=1)
            self.relu = nn.ReLU()
            self.maxPool = nn.MaxPool1d(kernel_size=3)
            self.flatten = nn.Flatten()
            self.lstm1 = nn.LSTM(input_size=1000,hidden_size=hiddenLayerSize, num_layers=numLayers,batch_first=True,dropout=0.1) # noof recurrent layers, num_layers is used to created stacked lstm. 
            self.fc1 = nn.Linear(in_features=hiddenLayerSize,out_features=outputSize)                            

    def forward(self,x,hidden=None):
        if self.model == "2lr" :
            lstm_out,hidden = self.lstm1(x,hidden)
            # print(lstm_out.shape)
            # lstm_out = lstm_out[:,-1,:]
            linear_out = self.fc1(lstm_out)
            # print(lstm_out[:,-1,:].shape , linear_out.shape) 
            return linear_out[:,-1,:],hidden #send only last one , predicted out
        
        if self.model == "smpl_aware" :
            r_input = x[:,:,:3] # first three inputs are global rotation
            t_input = x[:,:,-4:] # last four elemets are translation parameters
            p_input = x[:,:,3:-4] # all except first 3 and last 4 are pose parameters
            # print(p_input.shape)
            
            lstm_r_out , hidden_r = self.lstm_r(r_input)
            lstm_p_out ,hidden_p = self.lstm_p(p_input)
            lstm_t_out ,hidden_t = self.lstm_t(t_input)

            # r_out = self.fc_r(lstm_r_out[:,-1,:])
            # p_out = self.fc_p(lstm_p_out[:,-1,:])
            # t_out = self.fc_t(lstm_t_out[:,-1,:])

            out = torch.cat((lstm_r_out,lstm_p_out,lstm_t_out),dim=2)

            l_out = self.fc(out[:,-1,:].squeeze(1))
            return l_out,0

        elif self.model== "ann_2lr" :
            # lstm_out,hidden = self.lstm1(x,hidden)
            # lstm_out = lstm_out[:,-1,:]
            # fc1_out = self.fc1(lstm_out)
            # fc2_out = self.fc2(fc1_out)
            t_ = x.size(1) # get number of timestamps
            out = torch.tensor([])
            for i_ in range(t_) :
                fc1_out = self.fc1(x[:,i_,:])
                fc2_out = self.drop(self.relu(self.fc2(fc1_out)))
                fc3_out = self.fc3(fc2_out)
                fc3_out = self.maxPool(fc3_out.view(fc3_out.size(0),1,-1))
                fc3_out = fc3_out.view(fc3_out.size(0),-1)
                out = torch.cat((out,fc3_out),dim=1)
            out = out.view(x.size(0),t_,-1)
            lstm_out,hidden = self.lstm1(out,hidden)
            # lstm_out = lstm_out[:,-1,:]
            # fc3_out = self.drop(self.relu(self.fc3(lstm_out)))
            linear_out = self.fc4(lstm_out)
            return linear_out[:,-1,:],hidden #send only last one , predicted out
        
        elif self.model == "cnn_2lr":
            t_ = x.size(1) # get number of timestamps
            out = torch.tensor([])
            for i_ in range(t_) :
                cnn1_out = self.cnn1(x[:,i_,:].view(x.size(0),1,-1))
                cnn2_out = self.cnn2(cnn1_out)
                out_ = self.maxPool(cnn2_out)
                out_ = self.flatten(out_)
                out = torch.cat((out,out_),dim=1)
            out = out.view(x.size(0),t_,-1)
            lstm_out,hidden = self.lstm1(out,hidden)
            linear_out = self.fc1(lstm_out)
            return linear_out[:,-1,:],hidden
        
        elif self.model == "ann" :
            x = x.view(x.size(0),-1)
            fc1_out = self.drop(self.relu(self.fc1(x)))
            fc2_out = self.drop(self.relu(self.fc2(fc1_out)))
            fc3_out = self.drop(self.relu(self.fc3(fc2_out)))
            fc4_out = self.fc4(fc3_out)
            return fc4_out,0
    
def basicLSTMTrain(config,trainData,validData,typeOfLoss,model,device):

    hiddenLayerSize = config["hiddenLayerSize"]
    numLayers = config["numLayers"]
    learningRate = config["learningRate"]
    epochs = config["epochs"]
    batchSize = config["batchSize"]

    trainX , trainY = trainData
    validX, validY = validData

    inputSize = trainX.shape[2]
    outputSize = trainX.shape[2]

    # print(F"input shape {inputSize}")

    # convert them into tensors 
    trainX = torch.from_numpy(trainX).float().to(device)
    trainY = torch.from_numpy(trainY).float().to(device)

    # testX = torch.from_numpy(testData[:,:-1,:]).float()
    # testY = torch.from_numpy(testData[:,-1,:]).float()

    validX = torch.from_numpy(validX).float().to(device)
    validY = torch.from_numpy(validY).float().to(device)

    trainLoader = DataLoader(TensorDataset(trainX,trainY),batch_size=batchSize,drop_last=True) # make batches of train_data
    validLoader = DataLoader(TensorDataset(validX,validY),batch_size=batchSize,drop_last=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select GPU if available
    
    model = BasicLSTM(inputSize,hiddenLayerSize,outputSize,numLayers,model)
    model.to(device)
    # print(model)
    # return 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,factor=0.5,verbose=True)

    if typeOfLoss == "MSE" :
        criterion = torch.nn.MSELoss() # loss function
    elif typeOfLoss == "MAE" :
        criterion = torch.nn.L1Loss()

    # used to store losses for plotting
    vlArray = []
    tlArray = []

    validLossMin = float("inf") # used to check early stopping
    badEpoch = 0

    # Train and Eval loop 
    for epoch in range(epochs) :
        
        model.train() # switch to train model to train the network
        tl = [] # used to store training loss
        hidden_train = None

        # Training
        for i, data in enumerate(trainLoader) :

            _trainX , _trainY = data[0].to(device) , data[1].to(device)
            optimizer.zero_grad()

            predY, hidden_train = model(_trainX, hidden_train)
            hidden_train = None
            loss = criterion(predY,_trainY)
            loss.backward()
            optimizer.step()
            tl.append(loss.cpu().detach().item())
        
        # Eval
        vl = []
        hiddenValid = None
        model.eval() # change the model into eval mode 

        for _validX , _validY in validLoader :
            _validX = _validX.to(device)
            _validY = _validY.to(device)

            predY,hiddenValid = model(_validX,hiddenValid)
            hiddenValid = None # need to check why we need to do this
            # predY = predY[:,-1,:] # take last item as 
            loss = criterion(predY,_validY)
            vl.append(loss.cpu().detach().item())

        trainLoss = np.mean(tl)
        validLoss = np.mean(vl)

        tlArray.append(trainLoss)
        vlArray.append(validLoss)

        scheduler.step(validLoss) # ReduceLROnPlateau ref - https://pytorch.org/docs/stable/optim.html
        
        if epoch % 25 == 0 : # print for only every 10 epochs
            print(F"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} epoch {epoch}/{epochs} tl,vl => {trainLoss}, {validLoss}")

        # Early stopping to avoid overfitting
        if validLoss < validLossMin :
            validLossMin = validLoss # save current validLoss
            badEpoch = 0
            
            #save the model
            torch.save(model.state_dict(),config["modelName"]+".pth")
        else :
            if validLoss - validLossMin >= config["earlyStoppingDelta"] :
                badEpoch += 1 # increment badEpoch

            if badEpoch >= config["patience"] :
                print(F"Training stops in early due to overfitting in epoch {epoch}")
                print(F"tl Array {tlArray[-4:]} vl array {vlArray[-4:]} ")
                break # stop training
    del model
    torch.cuda.empty_cache()

    return {
        "trainingLoss" : tlArray,
        "validLoss" : vlArray
    }


    # if config["plotLoss"] :    
    #     # plotting the losses
    #     plt.figure()
    #     plt.plot(tlArray,label="Training")
    #     plt.plot(vlArray,label="Validation")
    #     plt.title(config["modelName"].replace(".pth",""))
    #     plt.legend()
    #     plt.show()

    # return tlArray , vlArray

def LCModelEval(config,testData,modelType,device) :
    testX , testY, testT_1 = testData
    testX = torch.from_numpy(testX).float().to(device)
    testY = torch.from_numpy(testY).float().to(device)
    testT_1 = torch.from_numpy(testT_1).float().to(device)
    # print(F"{testX.shape} , {testT_1.shape}")
    testLoader = DataLoader(TensorDataset(testX,testY,testT_1),batch_size=config["batchSize"],drop_last=True)

    inputSize = testX.shape[2]
    outputSize = testX.shape[2]
    hiddenLayerSize = config["hiddenLayerSize"]
    numLayers = config["numLayers"]

    model = BasicLSTM(inputSize,hiddenLayerSize,outputSize,numLayers,modelType)
    model.load_state_dict(torch.load(config["modelName"]+'.pth',map_location=device))
    model.to(device)

    gt = np.empty(shape=(0,76))
    pred = np.empty(shape=(0,76))

    model.eval()
    hiddenPredict = None
    # print(len(testLoader))
    for i, _data in enumerate(testLoader) :
        dataX = _data[0]
        dataY = _data[1].detach().cpu().numpy() # since dataLoader shuffles the data , we need to add testY to loader to get the both gt and pred in same order
        pose_t_1 = _data[2]
        # print(F"{i} => {dataX.size()}")
        predX , hiddenPredict = model(dataX,hiddenPredict)
        hiddenPredict = None
        # print(F"out shape => {predX.shape}, pose_t_t {pose_t_1.shape}")
        dt_pred = predX.detach().cpu().numpy()
        pose_t_1 = pose_t_1.detach().cpu().numpy()
        pose_pred = np.add(dt_pred,pose_t_1)
        pred = np.append(pred,pose_pred,axis=0)
        gt = np.append(gt,dataY,axis=0)
        # print(pred.shape)
    
    del model
    torch.cuda.empty_cache()
    return np.array(gt) , np.array(pred)


def basicLSTMEval(config,testData,modelType,device):
    """
    Evalutate Basic LSTM model
    
    params :
        config : config dict
        testData : data to be tested

    returns :
        gt -> ground truth SMPL data with shape [X:75] 72 SMPL pose + 3 translation parameters 
        pred -> predcited SMPL data with shape [X:75] 
    """
    
    # load test data
    testX , testY = testData

    testX = torch.from_numpy(testX).float().to(device)
    testY = torch.from_numpy(testY).float().to(device)

    testSet = TensorDataset(testX,testY)
    testLoader = DataLoader(testSet,batch_size=100,drop_last=True)

    inputSize = testX.shape[2]
    outputSize = testX.shape[2]
    hiddenLayerSize = config["hiddenLayerSize"]
    numLayers = config["numLayers"]

    model = BasicLSTM(inputSize,hiddenLayerSize,outputSize,numLayers,modelType)
    model.load_state_dict(torch.load(config["modelName"]+'.pth',map_location=device))
    model.to(device)

    gt = np.empty(shape=(0,76))
    pred = np.empty(shape=(0,76))

    model.eval()
    hiddenPredict = None

    for _data in testLoader :
        dataX = _data[0]
        dataY = _data[1]

        predX , hiddenPredict = model(dataX,hiddenPredict)
        hiddenPredict = None
        predX = predX.cpu().detach().numpy()
        pred = np.append(pred,predX,axis=0)
        gt = np.append(gt,dataY.cpu().detach().numpy(),axis=0)

    del model
    torch.cuda.empty_cache()

    return np.array(gt) , np.array(pred)

def getResults(gt,pred,shapeV,device, steps=None):

    model = smpl_torch.SMPLModel(device=device)

    print(F" number of values for test {pred.shape}")
    # due to memory constrains - we are processing this 1000 samples at a time
    if steps == None :
        vertex , joints , rotations = model(betas_, pose_, trans_)
    else :
        print(F"processing in steps for SMPL regession {gt.shape} , {pred.shape}")
        _samples = gt.shape[0]
        v_rmse = []
        mpjae = []
        mpjpe = []
        mpjae_without_root = []

        for _i in range(0,_samples,steps) :
            # vertex = [] # 6890 is vertex shape 
            # joints = [] # 24 joints 
            # rotations = [] # 24 rotations


            s_idx = _i
            e_idx = _i + steps

            # Running for pose only
            # pose_ = np.concatenate((gt[s_idx:e_idx,:-3], pred[s_idx:e_idx,:-3])) # add both so we can run SMPL model on them
            # trans_ = np.concatenate((gt[s_idx:e_idx,-3:], pred[s_idx:e_idx,-3:]))
            # betas_ = np.repeat(shapeV,pose_.shape[0],axis=0) # take random shape and make duplicates of it
            
            #all but last 4 values are pose. last 4 values are orig_cam parameters
            pose_ = np.concatenate((gt[s_idx:e_idx,:-4], pred[s_idx:e_idx,:-4])) # add both so we can run SMPL model on them
            trans_ = np.repeat(np.zeros(3),pose_.shape[0],axis=0)
            betas_ = np.repeat(shapeV,pose_.shape[0],axis=0) # take random shape and make duplicates of it

            pose_ = torch.from_numpy(pose_).type(torch.float64).to(device)
            trans_ = torch.from_numpy(trans_).type(torch.float64).to(device)
            betas_ = torch.from_numpy(betas_).type(torch.float64).to(device)
            _vertex, _joints, _rotations = model(betas_,pose_,trans_)
            # vertex.extend(_vertex)
            # joints.extend(_joints)
            # rotations.extend(_rotations)

            vertex = _vertex
            joints = _joints
            rotations = _rotations
            #print(vertex.shape, joints.shape, rotations.shape)

            vertex_gt,vertex_pred = np.vsplit(vertex,2)
            joints_gt, joints_pred = np.vsplit(joints,2)
            rotations_gt,rotations_pred = np.vsplit(rotations,2)

            # print(vertex_gt.shape,joints_pred.shape,rotations_gt.shape)
            # print(rotations_gt.shape)
            for i in range(vertex_gt.shape[0]) :
                # print(i)
                v_rmse.append(evalModel.vertexRMSE(vertex_gt[i,:,:],vertex_pred[i,:,:]))
                # print(rotations_gt[i,:,:,:].shape)
                mpjae.append(evalModel.MPJAE(rotations_gt[i,:,:,:],rotations_pred[i,:,:,:]))
                mpjpe.append(evalModel.MPJPE(joints_gt[i,:,:],joints_pred[i,:,:]))
                mpjae_without_root.append(evalModel.MPJAEWithoutRoot(rotations_gt[i,:,:,:],rotations_pred[i,:,:,:]))
            #print(F"len of v_rmse is {len(v_rmse)} , {v_rmse[0]}")
            # print(np.mean(mpjpe))
            # print(np.degrees(np.mean(mpjae)))
        gr_error = evalModel.getGlobalRotError(pred,gt)
        tr_error = evalModel.getTransError(pred,gt)
        print(F"vertex : {np.mean(v_rmse)*1000} , mpjpe : {np.mean(mpjpe)*1000} , mpjae : {np.degrees(np.mean(mpjae))} , \
        mpjae_without_root : {np.degrees(np.mean(mpjae_without_root))}, gr_error : {gr_error} , tr_error ; {tr_error}")
    
    del model
    torch.cuda.empty_cache()    
    
    return {
        "vertex" : np.mean(v_rmse) * 1000, # convert meters into mm
        "mpjpe" : np.mean(mpjpe) * 1000, # convert meters into mm
        "mpjae" : np.degrees(np.mean(mpjae)),
        "mpjae_without_root" : np.degrees(np.mean(mpjae_without_root)),
        "gr_error" : gr_error,
        "tr_error" : tr_error
    }


if __name__ == "__main__" :
    from LSTMAutoEncoder import *

    config = {
        "learningRate" : 0.0001,
        "hiddenLayerSize" : 100,
        "numLayers" : 6,
        "batchSize" : 50,
        "epochs" : 500,
        "plotLoss" : False,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "modelName" : "2LR-LSTM.pth",
        "dataFile" : "pedx_out.json",
        "lookBack" : 5,
        "key" : "pose_and_trans", # for which data model should run on
        "resultDir" : "results"
    }

    results = {}

    # 2LR-LSTM Model

    mData = ModelData(config)

    trainData = mData.getTrainData()
    validData = mData.getValidData()
    testData = mData.getTestData()

    print(F"Train data shape is {trainData[0].shape} , {trainData[1].shape}")

    tl_b,vl_b = basicLSTMTrain(config,trainData,validData,'basic','2lr')
    gt , pred = basicLSTMEval(config,testData,'2lr')
    r_2lr = getResults(gt,pred,steps=1000) # steps -> no of samples for SMPL regression at once ( we cant process all due to memory constrains)

    config["modelName"] = "autoEncoder"

    mData = ModelData(config)
    trainData = mData.getTrainData()
    validData = mData.getValidData()
    testData = mData.getTestData()
    
    m = lstm_seq2seq(trainData[0].shape[-1],config["hiddenLayerSize"])
    m.train(config,trainData,validData)
    gt , pred = m.test(config,testData)
    vae = getResults(gt,pred,steps=500) 

    print(r_2lr)
    print(vae)

    # config["modelName"] = "2LR-LSTM_ANN.pth"
    # # config["numLayers"] = 2
    # # config["patience"] = 50

    # mData = ModelData(config)

    # trainData = mData.getTrainData()
    # validData = mData.getValidData()
    # testData = mData.getTestData()

    # tl_b,vl_b = basicLSTMTrain(config,trainData,validData,'basic','ann_2lr')
    # gt , pred = basicLSTMEval(config,testData,'ann_2lr')
    # r_2lr_ann = getResults(gt,pred,steps=1000)

    # config["modelName"] = "ANN.pth"
    # # config["numLayers"] = 2
    # # config["patience"] = 50

    # mData = ModelData(config)

    # trainData = mData.getTrainData()
    # validData = mData.getValidData()
    # testData = mData.getTestData()

    # tl_b,vl_b = basicLSTMTrain(config,trainData,validData,'basic','ann')
    # gt , pred = basicLSTMEval(config,testData,'ann')
    # r_ann = getResults(gt,pred,steps=1000)

    # # LC model

    config["modelName"] = "LC.pth"
    # config["numLayers"] = 2

    mData = ModelData(config)

    lcTrainData = mData.getLcTrainData()
    lcValidData = mData.getLcValidData()
    lcTestData = mData.getLcTestData()


    tl_l,vl_l = basicLSTMTrain(config,lcTrainData,lcValidData,'lc','2lr')

    gt , pred = LCModelEval(config,lcTestData,'2lr')
    r_lc = getResults(gt,pred,steps=1000) 


    config["modelName"] = "LC-CNN.pth"
    # config["numLayers"] = 2

    mData = ModelData(config)

    lcTrainData = mData.getLcTrainData()
    lcValidData = mData.getLcValidData()
    lcTestData = mData.getLcTestData()


    tl_l,vl_l = basicLSTMTrain(config,lcTrainData,lcValidData,'lc','cnn_2lr')

    gt , pred = LCModelEval(config,lcTestData,'cnn_2lr')
    r_lc_cnn = getResults(gt,pred,steps=1000) 


    config["modelName"] = "LC-ANN.pth"
    # config["numLayers"] = 2

    mData = ModelData(config)

    lcTrainData = mData.getLcTrainData()
    lcValidData = mData.getLcValidData()
    lcTestData = mData.getLcTestData()


    tl_l,vl_l = basicLSTMTrain(config,lcTrainData,lcValidData,'lc','ann_2lr')

    gt , pred = LCModelEval(config,lcTestData,'ann_2lr')
    r_lc_ann = getResults(gt,pred,steps=1000) 



    print(F"r_2lr => {r_2lr}")
    print(F"r_2lr_ann => {r_2lr_ann}")
    print(F"r_ann => {r_ann}")
    print(F"r_lc => {r_lc}")
    print(F"r_lc_ann => {r_lc_ann}")
    print(F"r_lc_cnn => {r_lc_cnn}")

    # print(F"r_lc => {r_lc}")

    # # 2LR-LSTM with lookback 1 - i.e next frame prediction with l = 1

    # config["lookBack"] = 1
    # config["modelName"] = "2LR-LSTM_l1.pth"

    # mData = ModelData(config)

    # trainData = mData.getTrainData()
    # validData = mData.getValidData()
    # testData = mData.getTestData()

    # tl_b_l1,vl_b_l1 = basicLSTMTrain(config,trainData,validData,'basic')
    # gt , pred = basicLSTMEval(config,testData)
    # r_2lr_l1 = getResults(gt,pred)

    # # LC with lookback 1

    # config["modelName"] = "LC_l1.pth"
    # config["numLayers"] = 2
    # config["hiddenLayerSize"] = 32
    # config["lookBack"] = 2

    # mData = ModelData(config)

    # lcTrainData = mData.getLcTrainData()
    # lcValidData = mData.getLcValidData()
    # lcTestData = mData.getLcTestData()


    # tl_l_l1,vl_l_l1 = basicLSTMTrain(config,lcTrainData,lcValidData,'lc')

    # gt , pred = LCModelEval(config,lcTestData)
    # r_lc_l1 = getResults(gt,pred) 

    # with open("result.json","w") as fd :
    #     json.dump({
    #         "2LR" : r_2lr,
    #         "LC" : r_lc,
    #         "2LR_l1" : r_2lr_l1,
    #         "LC_l1" : r_lc_l1
    #     },fd)


    # plt.figure(1)
    # plt.plot(tl_b,label="2LR-Training")
    # plt.plot(vl_b,label="2LR-Validation")
    # plt.legend()
    # plt.title("2LR-LSTM Training")
    # plt.savefig(os.path.join(config["resultDir"],"2LR-Training_Losses.png"))

    # plt.figure(2)
    # plt.plot(tl_l,label="Lc-Training")
    # plt.plot(vl_l,label="Lc-Validation")
    # plt.title("LC Model Training")
    # plt.legend()
    # plt.savefig(os.path.join(config["resultDir"],"LC-Training_Losses.png"))

    # plt.figure(3)
    # plt.plot(tl_b_l1,label="2LR-l1 Training")
    # plt.plot(vl_b_l1,label="2LR-l1 Validation")
    # plt.legend()
    # plt.title("2LR-LSTM Training with l=1")
    # plt.savefig(os.path.join(config["resultDir"],"2LR_l1-Training_Losses.png"))

    # plt.figure(4)
    # plt.plot(tl_l_l1,label="Lc_l1 Training")
    # plt.plot(vl_l_l1,label="Lc_l1 Validation")
    # plt.title("LC Model Training with l=1")
    # plt.legend()
    # plt.savefig(os.path.join(config["resultDir"],"LC_l1-Training_Losses.png"))
    # # plt.show()

    
