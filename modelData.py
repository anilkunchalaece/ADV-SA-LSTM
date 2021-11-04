"""
Author : Kunchala Anil
This script will contains the data for model training , validation and testing
"""
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import preprocess 
import numpy as np

class ModelData():
    def __init__(self,config) :
        self.data = preprocess.loadData2(config["dataFileDir"],config["lookBack"],config["key"])
        self.lr2LstmData = self.splitDataFor2LR_LSTM()
        self.lcData = self.splitDataForLc()

    def splitDataFor2LR_LSTM(self) :
        self.trainAndValidData , self.testData = train_test_split(self.data,test_size=0.1,shuffle=True)
        self.trainData, self.validData = train_test_split(self.trainAndValidData,test_size=0.1,shuffle=True) 
        return {
            "train" : self.trainData, 
            "valid" : self.validData, 
            "test" : self.testData
        }

    def splitDataForLc(self) :
        """
        for Lc we use difference as training data
        """
        # self.LcData = np.diff(self.data,axis=0)
        # trainAndValidData , testData = train_test_split(self.data,test_size=0.06,shuffle=True)
        # trainData, validData = train_test_split(trainAndValidData,test_size=0.1,shuffle=True) 

        return {
            "train" : np.diff(self.trainData,axis=1), 
            "valid" : np.diff(self.validData,axis=1), 
            "test" : (np.diff(self.testData,axis=1) , self.testData) # to calculate the erros we need both diff and original data
        }               

    def getTrainData(self):
        self.trainX = self.lr2LstmData["train"][:,:-1,:] # get all except last item - training labels
        self.trainY = self.lr2LstmData["train"][:,-1,:] # get last item - output label
        # print(F"Train data shape is {self.trainX.shape} , {self.trainY.shape}")
        return self.trainX, self.trainY
    
    def getValidData(self):
        self.validX = self.lr2LstmData["valid"][:,:-1,:]
        self.validY = self.lr2LstmData["valid"][:,-1,:]
        return self.validX, self.validY
    
    def getTestData(self):
        self.testX = self.lr2LstmData["test"][:,:-1,:]
        self.testY = self.lr2LstmData["test"][:,-1,:]
        return self.testX , self.testY

    def getLcTrainData(self) :
        self.trainX = self.lcData["train"][:,:-1,:] # get all except last item - training labels
        self.trainY = self.lcData["train"][:,-1,:] # get last item - output label
        return self.trainX, self.trainY

    def getLcValidData(self) :
        self.validX = self.lcData["valid"][:,:-1,:]
        self.validY = self.lcData["valid"][:,-1,:]
        return self.validX, self.validY

    def getLcTestData(self):
        self.testX = self.lcData["test"][0][:,:-1,:] # diff is the test
        self.testY = self.lcData["test"][1][:,-1,:] # testY is the original last pose
        self.testT_1 = self.lcData["test"][1][:,-2,:]# pose data at T-1 used to calculate predicted pose using diff - x_t = x_t-1 + dt
        return self.testX , self.testY , self.testT_1


if __name__ == "__main__" :
    config = {
        "dataFile" : "out.json",
        "lookBack" : 5,
        "key" : "pose_and_trans"
    }
    mData = ModelData(config)
    print(mData.getLcTestData()[0].shape)