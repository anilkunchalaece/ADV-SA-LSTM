"""
Adversarial SMPL Aware 2LR-LSTM
using frame diff and mae loss function
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import json 
import matplotlib.pyplot as plt
from torchModels import getResults
from torch.autograd import Variable
import datetime
#torch.backends.cudnn.enabled = False

class Generator(nn.Module) :
    def __init__(self,config):
        super(Generator,self).__init__()
        self.lstm_r = nn.LSTM(input_size=3,hidden_size=2,num_layers=config["num_layers"],batch_first=True)
        self.lstm_p = nn.LSTM(input_size=69,hidden_size=8,num_layers=config["num_layers"],batch_first=True)
        self.lstm_t = nn.LSTM(input_size=4,hidden_size=2,num_layers=config["num_layers"],batch_first=True)
        # self.fc_r = nn.Linear(in_features=2,out_features=3)
        # self.fc_p = nn.Linear(in_features=8,out_features=69)
        # self.fc_t = nn.Linear(in_features=2,out_features=4)
        self.fc = nn.Linear(in_features=12,out_features=76)
    
    def forward(self,x) :
        r_input = x[:,:,:3] # first three inputs are global rotation
        t_input = x[:,:,-4:] # last four elemets are translation parameters
        p_input = x[:,:,3:-4] # all except first 3 and last 4 are pose parameters
    
        lstm_r_out , hidden_r = self.lstm_r(r_input)
        lstm_p_out ,hidden_p = self.lstm_p(p_input)
        lstm_t_out ,hidden_t = self.lstm_t(t_input)

        out = torch.cat((lstm_r_out,lstm_p_out,lstm_t_out),dim=2)

        l_out = self.fc(out[:,-1,:].squeeze(1))
        return l_out

class Descriminator(nn.Module):
    def __init__(self,config):
        super(Descriminator,self).__init__()
        self.fc1 = nn.Linear(in_features=76,out_features=250)
        self.fc5 = nn.Linear(in_features=250,out_features=1)
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x_input) :
        fc1_out = self.tanh(self.dropout(self.fc1(x_input)))
        output = self.sigmoid(self.fc5(fc1_out))
        return output


class Descriminator2(nn.Module):
    def __init__(self,config):
        super(Descriminator2,self).__init__()
        self.fc1 = nn.Linear(in_features=config["des_fc1_in"],out_features=config["des_fc1_out"])
        self.fc2 = nn.Linear(in_features=config["des_fc2_in"],out_features=config["des_fc2_out"])
        self.fc3 = nn.Linear(in_features=config["des_fc3_in"],out_features=config["des_fc3_out"])
        self.fc4 = nn.Linear(in_features=config["des_fc4_in"],out_features=config["des_fc4_out"])
        self.fc5 = nn.Linear(in_features=config["des_fc5_in"],out_features=config["des_fc5_out"])
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x_input) :
        fc1_out = self.tanh(self.dropout(self.fc1(x_input)))
        fc2_out = self.tanh(self.dropout(self.fc2(fc1_out)))
        fc3_out = self.tanh(self.dropout(self.fc3(fc2_out)))
        fc4_out = self.tanh(self.dropout(self.fc4(fc3_out)))
        output = self.sigmoid(self.fc5(fc4_out))
        return output



class AdversarialSMPLAwareLSTM :
    def __init__(self) :
        super(AdversarialSMPLAwareLSTM,self).__init__()
        self.config = self.getConfig()
        self.des = Descriminator(self.config)
        self.gen = Generator(self.config)

    def getConfig(self) :
        """
        This config is used to define network i.e - generator and descriminiator
        For generator we use SMPLAwareAutoEncoder - Network config in regards to that will be found in the SMPLAwareAutoEncoder.py
        """
        return {
            "des_fc1_in" : 76,
            "des_fc1_out" : 250,
            "des_fc2_in" : 250,
            "des_fc2_out" : 500,
            "des_fc3_in" : 500,
            "des_fc3_out" : 250,
            "des_fc4_in" : 250,
            "des_fc4_out" : 50,
            "des_fc5_in" : 50,
            "des_fc5_out" : 1,
            "gen_input_size" : 76,
            "gen_num_lstm_layers" : 2,
            "hiddenLayerSize" : 32,
            "num_layers" : 2
        }

    def train(self,config,trainData,lossType,device) :
        learningRate_gen = config["learningRate_gen"]
        learningRate_des = config["learningRate_des"]
        epochs = config["epochs"]
        batchSize = config["batchSize"]

        trainX , trainY = trainData
        trainX = torch.from_numpy(trainX).float().to(device)
        trainY = torch.from_numpy(trainY).float().to(device)

        trainLoader = DataLoader(TensorDataset(trainX,trainY),batch_size=batchSize,drop_last=True) # make batches of train_data

        optimizer_gen = optim.Adam(self.gen.parameters(), lr = learningRate_gen)
        optimizer_des = optim.Adam(self.des.parameters(), lr = learningRate_des)

        #Checke LSGAN - https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/
        criterion_gen = torch.nn.MSELoss()
        criterion_des = torch.nn.MSELoss()
        
        # we will test both MSE and MAE as a loss function
        if lossType == "MSE" :
            criterion_smpl_lr = torch.nn.MSELoss() # used in addition to the gen_loss
        elif lossType == "MAE" :
            criterion_smpl_lr = torch.nn.L1Loss() # MAE loss
        else :
            raise("Invalid loss type ! Please check loss type")

        self.gen = self.gen.to(device)
        self.des = self.des.to(device)
        
        loss_total = {
            "gl_total" : [],
            "dl_total" : [],
            "gl_adv" : [],
            "gl_ae" : [],
            "dl_fake" : [],
            "dl_real" : []
        }

        for epoch in range(epochs) :
            _gl_total = []
            _dl_total = []
            _dl_fake = []
            _dl_real = []
            _gl_adv = []
            _gl_ae = []

            for i, data in enumerate(trainLoader) :
                _trainX , _trainY = data[0].to(device) , data[1].to(device)
                # _trainX in 'n' number of past poses for which _trainY is predicted pose. 
                # So for descriminator we should ignore _trainX and consider only _trainY 
                
                ## Trianing Descriminator
                # Traing Descriminator on real data
                optimizer_des.zero_grad()

                               
                self.des.train()
                self.gen.eval()

                real_pred = self.des(_trainY)
                # real_targets = torch.from_numpy(np.ones((_trainY.size(0),1))).float().to(device)
                real_targets = torch.ones((_trainY.size(0),1),dtype=torch.float,requires_grad=True).to(device)
                

                # Training descriminator on data from generator
                fake_data = self.gen(_trainX)
                fake_pred = self.des(fake_data)
                # fake_targets = torch.from_numpy(np.zeros((_trainY.size(0),1))).float().to(device)
                fake_targets = torch.ones((_trainY.size(0),1),dtype=torch.float,requires_grad=True).to(device)
                
                real_loss = torch.mul(criterion_des(real_pred,real_targets),0.5).to(device)
                fake_loss = torch.mul(criterion_des(fake_pred,fake_targets),0.5).to(device)

                des_loss = torch.add(real_loss, fake_loss).to(device)
                des_loss.backward()                
                optimizer_des.step()
                
                # training Generator
                optimizer_gen.zero_grad()
                self.gen.train()
                self.des.eval()

                #generate data using generator
                fake_data = self.gen(_trainX)
                pred = self.des(fake_data)
                # targets = torch.from_numpy(np.ones((_trainY.size(0),1))).float().to(device) # for training generator , we need to force pred as real outputs
                targets = torch.zeros((_trainY.size(0),1),dtype=torch.float,requires_grad=True).to(device)

                gen_loss = torch.mul(criterion_gen(pred,targets),0.5).to(device)
                smpl_lstm_lr_loss = criterion_smpl_lr(_trainY,fake_data)

                gen_loss = torch.add(torch.mul(gen_loss,0.40).to(device), smpl_lstm_lr_loss).to(device)
                gen_loss.backward()
                optimizer_gen.step()

                _gl_total.append(gen_loss.cpu().detach().item())
                _dl_total.append(des_loss.cpu().detach().item())

                _gl_adv.append(gen_loss.cpu().detach().item())
                _gl_ae.append(smpl_lstm_lr_loss.cpu().detach().item())

                _dl_fake.append(fake_loss.cpu().detach().item())
                _dl_real.append(real_loss.cpu().detach().item())

                # del targets,fake_data,fake_targets,fake_pred,pred,real_targets
                # torch.cuda.empty_cache()
            
            
            if epoch % 25 == 0 :
                print(F"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} epoch {epoch}/{epochs} , DL_T {np.mean(_dl_total)} , DL_F {np.mean(_dl_fake)}, DL_R {np.mean(_dl_real)}, GL_T {np.mean(_gl_total)}, GL_Adv {np.mean(_gl_adv)}, GL_AE {np.mean(_gl_ae)}")
            
            loss_total["dl_total"].append(np.mean(_dl_total))
            loss_total["dl_real"].append(np.mean(_dl_real))
            loss_total["dl_fake"].append(np.mean(_dl_fake))
            loss_total["gl_total"].append(np.mean(_gl_total))
            loss_total["gl_adv"].append(np.mean(_gl_adv))
            loss_total["gl_ae"].append(np.mean(_gl_ae))
        # plt.plot(loss_total["dl_total"],label="DL_Total")
        # plt.plot(loss_total["dl_real"],label="DL_Real")
        # plt.plot(loss_total["dl_fake"],label="DL_Fake")
        # plt.legend()
        # plt.title("Adversarial learning-DL")
        # plt.show()
        # plt.plot(loss_total["gl_total"],label="GL_Total")
        # plt.plot(loss_total["gl_adv"],label="GL_ADV")
        # plt.plot(loss_total["gl_ae"],label="GL_AE")
        # plt.legend()
        # plt.title("Adversarial learning-GL")
        # plt.show()

        torch.save(self.gen.state_dict(),config["modelName"]+".pth")
        # with open("advLoss.json",'w') as fd :
        #     json.dump(loss_total,fd)
        return loss_total

    def test(self,config,testData,inputType,device) :
        if inputType == "frame_diff" :
            testX , testY, testT_1 = testData
            testX = torch.from_numpy(testX).float().to(device)
            testY = torch.from_numpy(testY).float().to(device)
            testT_1 = torch.from_numpy(testT_1).float().to(device)
            # print(F"{testX.shape} , {testT_1.shape}")
            testLoader = DataLoader(TensorDataset(testX,testY,testT_1),batch_size=config["batchSize"],drop_last=True)

            model = self.gen
            model.load_state_dict(torch.load(config["modelName"]+'.pth',map_location=device))
            model.to(device)

            gt = np.empty(shape=(0,76))
            pred = np.empty(shape=(0,76))

            model.eval()

            # print(len(testLoader))
            for i, _data in enumerate(testLoader) :
                dataX = _data[0]
                dataY = _data[1].detach().cpu().numpy() # since dataLoader shuffles the data , we need to add testY to loader to get the both gt and pred in same order
                pose_t_1 = _data[2]
                # print(F"{i} => {dataX.size()}")
                predX = model(dataX)
                # print(F"out shape => {predX.shape}, pose_t_t {pose_t_1.shape}")
                dt_pred = predX.detach().cpu().numpy()
                pose_t_1 = pose_t_1.detach().cpu().numpy()
                pose_pred = np.add(dt_pred,pose_t_1)
                pred = np.append(pred,pose_pred,axis=0)
                gt = np.append(gt,dataY,axis=0)
                # print(pred.shape)
        
        elif inputType == "pose" :    
            # load test data
            testX , testY = testData
            testX = torch.from_numpy(testX).float().to(device)
            testY = torch.from_numpy(testY).float().to(device)

            testSet = TensorDataset(testX,testY)
            testLoader = DataLoader(testSet,batch_size=config["batchSize"],drop_last=True)

            model = self.gen
            model.load_state_dict(torch.load(config["modelName"]+'.pth',map_location=device))
            model.to(device)

            gt = np.empty(shape=(0,76))
            pred = np.empty(shape=(0,76))

            model.eval()

            for _data in testLoader :
                dataX = _data[0]
                dataY = _data[1]

                predX  = model(dataX)
                predX = predX.cpu().detach().numpy()
                pred = np.append(pred,predX,axis=0)
                gt = np.append(gt,dataY.cpu().detach().numpy(),axis=0)

        del model
        torch.cuda.empty_cache()

        return np.array(gt) , np.array(pred)



if __name__ == "__main__" :
    from modelData import ModelData # get training data
    config = {
        "learningRate_gen" : 0.0002,
        "learningRate_des" : 0.0006,
        "batchSize" : 50,
        "epochs" : 500,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "modelName" : "adversarialSMPLNetwork",
        "dataFileDir" : "data/pedx",
        "lookBack" : 5,
        "key" : "pose_orig_cam", # for which data model should run on
        "resultDir" : "results"
    }

    lossType = "MAE"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    mData = ModelData(config)
    trainData = mData.getLcTrainData()
    validData = mData.getLcValidData()
    testData = mData.getLcTestData()


    adv = AdversarialSMPLAwareLSTM()
    adv.train(config,trainData,lossType,device)   
    # gt,pred = adv.test(config,testData,device)
    # shapeV = ((np.random.rand(1,10) - 0.5)*0.06) # random shape value used for all models for testing
    # adv_tr = getResults(gt,pred,shapeV,device=None,steps=200) 
    # print(adv_tr)



