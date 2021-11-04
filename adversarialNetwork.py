"""
This script is used to train network in adversarial manner
AutoEncoder is used as generator and recurrent descrimanator will used to train the generator in adversarial fashion to improve
the generator (AutoEncoder) accuracy (?) and force generator to generate natural plausible poses

ref : https://towardsdatascience.com/getting-started-with-gans-using-pytorch-78e7c22a14a5
"""

from numpy.lib.function_base import append
import torch
from torch.functional import tensordot
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import json

from torchModels import getResults
import SMPLAwareAutoEncoder
import LSTMAutoEncoder
import matplotlib.pyplot as plt


class Descriminator(nn.Module) :

    def __init__(self,config):
        super(Descriminator,self).__init__()
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

class Generator(nn.Module) :
    def __init__(self,config,genType):
        super(Generator,self).__init__()
        if genType == "smpl_ae" :
            self.ae = SMPLAwareAutoEncoder.SMPLAwareAE(config["gen_input_size"],config["gen_num_lstm_layers"])
        elif genType == "ae" :
            self.ae = LSTMAutoEncoder.lstm_seq2seq(config["gen_input_size"],config["hiddenLayerSize"],config["gen_num_lstm_layers"])

class AdversarialNetwork(nn.Module) :
    def __init__(self,genType,inputType):
        """
        inputType : whether we are using pose data or frame difference data
        -- pose , frame_diff
        """
        super(AdversarialNetwork,self).__init__()
        self.config = self.getConfig()
        self.des = Descriminator(self.config)
        self.gen = Generator(self.config,genType)
        self.inputType = inputType
    
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
            "hiddenLayerSize" : 32
        }

    def getOutputFromGenerator(self,testX,enc_obj,dec_obj) :
        """
        For given data, generate output i.e future pose using past poses
        """
        decoder_input = testX[:,-1,:] # get last element
        decoder_input = decoder_input.unsqueeze(dim=1) # add extra dimesion so it will be suitable for lstm input

        # put both encoder and decoder in eval mode. Do I have to do that ?
        enc_obj.eval()
        dec_obj.eval()

        enc_output , enc_hidden = enc_obj(testX)

        dec_output = dec_obj(decoder_input,enc_hidden) # we are passing the enc_hidden state to dec
        return dec_output.squeeze(dim=1)
    
    def train(self,config,trainData,lossType) :
        """
        Traning both generator and descrimniator adversarily
        config - used to define hyper parameters for training
        """

        learningRate_gen = config["learningRate_gen"]
        learningRate_des = config["learningRate_des"]
        epochs = config["epochs"]
        batchSize = config["batchSize"]

        trainX , trainY = trainData
        trainX = torch.from_numpy(trainX).float()
        trainY = torch.from_numpy(trainY).float()

        trainLoader = DataLoader(TensorDataset(trainX,trainY),batch_size=batchSize,drop_last=True) # make batches of train_data

        optimizer_gen = optim.Adam(self.gen.parameters(), lr = learningRate_gen)
        optimizer_des = optim.Adam(self.des.parameters(), lr = learningRate_des)
        
        # check the parameters names -> to see if generator i.e SMPLAwareAutoEncoder initliazed properly or not
        # for name, param in self.gen.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        #Checke LSGAN - https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/
        criterion_gen = torch.nn.MSELoss()
        criterion_des = torch.nn.MSELoss()
        
        # we will test both MSE and MAE as a loss function
        if lossType == "MSE" :
            criterion_ae = torch.nn.MSELoss() # used in addition to the gen_loss
        elif lossType == "MAE" :
            criterion_ae = torch.nn.L1Loss() # MAE loss
        else :
            raise("Invalid loss type ! Please check loss type")
        
        loss_total = {
            "gl_total" : [],
            "dl_total" : [],
            "gl_adv" : [],
            "gl_ae" : [],
            "dl_fake" : [],
            "dl_real" : []
        }

        for epoch in range(epochs) :
            # print(F"running epoch {epoch}")
            _gl_total = []
            _dl_total = []
            _dl_fake = []
            _dl_real = []
            _gl_adv = []
            _gl_ae = []

            for i, data in enumerate(trainLoader) :
                _trainX , _trainY = data[0] , data[1]

                # _trainX in 'n' number of past poses for which _trainY is predicted pose. 
                # So for descriminator we should ignore _trainX and consider only _trainY 
                
                ## Trianing Descriminator
                # Traing Descriminator on real data
                optimizer_des.zero_grad()
                self.gen.ae.encoder.eval()
                self.gen.ae.decoder.eval()
                self.des.train()

                real_pred = self.des(_trainY)
                real_targets = torch.ones(_trainY.size(0),1)
                real_loss = 0.5*criterion_des(real_pred,real_targets)

                # Training Descriminator on data from generator
                fake_data = self.getOutputFromGenerator(_trainX,self.gen.ae.encoder,self.gen.ae.decoder) # use generator to generate fake data
                fake_pred = self.des(fake_data)
                fake_targets = torch.zeros(_trainY.size(0),1)
                fake_loss = 0.5*criterion_des(fake_pred,fake_targets)

                des_loss = real_loss + fake_loss # check LS-GAN
                des_loss.backward()
                optimizer_des.step()
                
                ## Training Generator
                optimizer_gen.zero_grad()
                self.gen.ae.encoder.train()
                self.gen.ae.decoder.train()
                self.des.eval()

                # generate fake images
                fake_data = self.getOutputFromGenerator(_trainX,self.gen.ae.encoder,self.gen.ae.decoder) # use generator to generate fake data
                pred = self.des(fake_data)
                targets = torch.ones(_trainY.size(0),1) # note while training we use 1's as lables incontrast to descrminator
                # i.e we force generator weights to fool descriminator
                gen_loss = 0.5 * criterion_gen(pred,targets)

                ae_loss = criterion_ae(_trainY,fake_data)

                g_loss = (0.15*gen_loss) + ae_loss

                g_loss.backward()
                optimizer_gen.step()

                _gl_total.append(g_loss.item())
                _dl_total.append(des_loss.item())

                _gl_adv.append(gen_loss.item())
                _gl_ae.append(ae_loss.item())

                _dl_fake.append(fake_loss.item())
                _dl_real.append(real_loss.item())
            
            
            if epoch % 30 == 0 :
                print(F"epoch {epoch}/{epochs} , DL_T {np.mean(_dl_total)} , DL_F {np.mean(_dl_fake)}, DL_R {np.mean(_dl_real)}, GL_T {np.mean(_gl_total)}, GL_Adv {np.mean(_gl_adv)}, GL_AE {np.mean(_gl_ae)}")
            
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

        torch.save(self.gen.ae.encoder.state_dict(),config["modelName"]+self.inputType+"_enc.pth")
        torch.save(self.gen.ae.decoder.state_dict(),config["modelName"]+self.inputType+"_dec.pth")
        # with open("advLoss.json",'w') as fd :
        #     json.dump(loss_total,fd)
        return loss_total

    def test(self,config,testData) :
        # load test data
        if self.inputType == "pose" :
            testX , testY = testData
            testX = torch.from_numpy(testX).float()
            testY = torch.from_numpy(testY).float()
            gt = testY
        elif self.inputType == "frame_diff" :
            testX, testY_org ,testT_1 = testData
            testX = torch.from_numpy(testX).float()
            testY = torch.from_numpy(testT_1).float() # last time step value to calculate the next one using diff pose data at T-1 used to calculate predicted pose using diff - x_t = x_t-1 + dt
            gt = testY_org # check modelData.py for more details

        testLoader = DataLoader(TensorDataset(testX,testY),batch_size=1)

        self.gen.ae.encoder.load_state_dict(torch.load(config["modelName"]+self.inputType+"_enc.pth"))
        self.gen.ae.decoder.load_state_dict(torch.load(config["modelName"]+self.inputType+"_dec.pth"))

        
        pred = torch.tensor([])

        self.gen.ae.encoder.eval()
        self.gen.ae.decoder.eval()

        for _testX , _testY in testLoader :
            if len(_testY.size()) == 2 :
                _testY = _testY.view(_testX.size(0),-1,_testX.size(-1)) # convert 2D to 3D used for single time step prediction

            encoder_output_val, encoder_hidden_val = self.gen.ae.encoder(_testX)
            
            # initialize tensor for predictions
            outputs_val = torch.zeros(_testY.size(0),_testY.size(1), _testY.size(-1))

            # decode input_tensor
            decoder_input_val = _testX[:,-1, :].view(_testX.size(0),-1,_testX.size(-1))#the value used as input to decoder i.e last time step
            decoder_hidden_val = encoder_hidden_val
            
            for t in range(_testY.size(1)):
                decoder_output_val = self.gen.ae.decoder(decoder_input_val, decoder_hidden_val)
                if self.inputType == "frame_diff" :
                    # print(decoder_output_val.shape)
                    # print(_testY[:,t,:].shape)
                    decoder_output_val = decoder_output_val.squeeze(dim=1) + _testY[:,t,:]
                outputs_val[:,t,:] = decoder_output_val.view(decoder_output_val.size(0),-1) 
                decoder_input_val = decoder_output_val
            
            pred = torch.cat((pred,outputs_val),dim=1) 
        
        return np.array(gt) , pred.view(-1,pred.size(-1)).detach().numpy()

if __name__ == "__main__" :
    from modelData import ModelData # get training data
    config = {
        "learningRate_gen" : 0.0002,
        "learningRate_des" : 0.0006,
        "batchSize" : 50,
        "epochs" : 5000,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "modelName" : "adversarialNetwork",
        "dataFile" : "modified_pedx_out.json",
        "lookBack" : 5,
        "key" : "pose_and_orig_cam", # for which data model should run on
        "resultDir" : "results"
    }
    
    lossType = "MSE"

    mData = ModelData(config)
    trainData = mData.getTrainData()
    validData = mData.getValidData()
    testData = mData.getTestData()

    adv = AdversarialNetwork()
    adv.train(config,trainData,lossType)
    gt,pred = adv.test(config,testData)
    shapeV = ((np.random.rand(1,10) - 0.5)*0.06) # random shape value used for all models for testing
    adv_tr = getResults(gt,pred,shapeV,steps=200) 
    print(adv_tr)