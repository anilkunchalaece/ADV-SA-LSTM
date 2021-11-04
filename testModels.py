"""
This script is used to test models for data
"""

from os import pipe2

from torch.nn.modules import loss
from LSTMAutoEncoder import *
from torchModels import *
from SMPLAwareAutoEncoder import *
from adversarialNetwork import *
import torch
from adversarialSMPLAwareLSTM import *
import gc
torch.manual_seed(42)

model_overhead = "t_3"

def runModels(config,trainData,validData,testData,lossType,datasetName,inputType,lookBack,device,nTest):
    """
    lossType -> basic or lc
    """
    shapeV = ((np.random.rand(1,10) - 0.5)*0.06) # random shape value used for all models for testing

    # #2LR-LSTM
    print(F"\n ~~~~~ Running 2LR-LSTM loss : {lossType}, dataset : {datasetName}, inputType : {inputType}, lookback {lookBack}~~~~~~~~~~~ \n")
    config["modelName"] = F"2LR_LSTM_{datasetName}_{lossType}_{inputType}_lb{lookBack}_t{nTest}"
    config["numLayers"] = 2
    loss = basicLSTMTrain(config,trainData,validData,lossType,'2lr',device)
    saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save loss in a file
    if inputType == 'pose' :
        gt , pred = basicLSTMEval(config,testData,'2lr',device)
        print(F"{gt.shape} , {pred.shape}")
    elif inputType == 'frame_diff' :
        gt, pred = LCModelEval(config,testData,'2lr',device)
        print(F"{gt.shape} , {pred.shape}")
    saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
    r_2lr = getResults(gt,pred,shapeV,device=None,steps=config["smplBatchSize"])

    # # # SMPL AWARE 2LR-LSTM
    if lossType == "MAE" and inputType == "frame_diff" : # run this only for frame_diff and MAE
        print(F"\n ~~~~~ Running SMPL AWARE 2LR-LSTM : {lossType}, dataset : {datasetName}, inputType : {inputType}, lookback {lookBack} ~~~~~~~~~~~ \n")
        config["modelName"] = F"SMPL_AWARE_LR_LSTM_{datasetName}_{lossType}_{inputType}_lb{lookBack}_t{nTest}"
        if inputType == "pose" :
            config["numLayers"] = 1
            if lossType == "MSE" :
                config["learningRate"] = 0.008
            elif lossType == "MAE" :
                config["learningRate"] = 0.004
        elif inputType == "frame_diff" :
            config["numLayers"] = 1
            if lossType == "MSE" :
                config["learningRate"] = 0.0008
            elif lossType == "MAE" :
                config["learningRate"] = 0.0001

        loss = basicLSTMTrain(config,trainData,validData,lossType,'smpl_aware',device)
        saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save loss in a file

        if inputType == 'pose' :
            gt , pred = basicLSTMEval(config,testData,'smpl_aware',device)
        elif inputType == 'frame_diff' :
            gt, pred = LCModelEval(config,testData,'smpl_aware',device)

        saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
        r_2lr_smpl = getResults(gt,pred,shapeV,device=None,steps=config["smplBatchSize"])
    else :
        r_2lr_smpl = 0

    
    if lossType == "MAE" and inputType == "frame_diff" : # run this only for frame_diff and MAE
        # ### ADV SMPL Aware Network
        print(F"\n ~~~~~ Running ADV SMPL AWARE 2LR-LSTM : {lossType}, dataset : {datasetName}, inputType : {inputType}, lookback {lookBack}~~~~~~~~~~~ \n")
        config["modelName"] = F"ADV_SMPL_AWARE_LR_LSTM_{datasetName}_{lossType}_lb{lookBack}_t{nTest}"
        config["learningRate_gen"] = 0.0001
        config["learningRate_des"] = 0.0006
        adv = AdversarialSMPLAwareLSTM()
        adv.train(config,trainData,lossType,device=torch.device('cpu'))   
        gt,pred = adv.test(config,testData,inputType,device=torch.device('cpu'))
        saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
        r_adv_2lr_smpl = getResults(gt,pred,shapeV,device=None,steps=config["smplBatchSize"]) 
    else :
        r_adv_2lr_smpl = 0

    # # #LSTM-VAE
    # print(F"\n ~~~~~ Running LSTM_AE {lossType} {datasetName} ~~~~~~~~~~~ \n")
    # config["modelName"] = F"LSTM_AE_{datasetName}_{lossType}"
    # config["numLayers"] = 2
    # config["learningRate"] = 0.0001
    # m = lstm_seq2seq(trainData[0].shape[-1],config["hiddenLayerSize"],config["numLayers"])
    # loss = m.train(config,trainData,validData,lossType)
    # saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save loss in a file
    # gt , pred = m.test(config,testData)
    # saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
    # vae = getResults(gt,pred,shapeV,steps=config["smplBatchSize"])
    # # vertex : 631.8405768257918 , mpjpe : 419.0733145715228 , mpjae : 8.245045440888402

    # ## SMPLAware AutoEncoder
    # print(F"\n ~~~~~ Running SMPL_AE ~~~~~~~~~~~ {lossType} {datasetName} \n")
    # config["modelName"] = F"SMPL_AE_{datasetName}_{lossType}"
    # config["learningRate"] = 0.001
    # config["numLayers"] = 2
    # m = SMPLAwareAE(trainData[0].shape[-1],config["numLayers"])
    # loss = m.train(config,trainData,validData,lossType)
    # saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save loss in a file
    # gt , pred = m.test(config,testData)
    # saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
    # smpl_ae = getResults(gt,pred,shapeV,steps=config["smplBatchSize"])

    # print(F"\n ~~~~~ Running ADV_AE {lossType} {datasetName} {inputType} ~~~~~~~~~~~ \n")
    # config["modelName"] = F"ADV_AE_{datasetName}_{lossType}"
    # config["learningRate_des"] = 0.0001
    # config["learningRate_gen"] = 0.0004
    # m = AdversarialNetwork(genType="ae",inputType=inputType)
    # loss = m.train(config,trainData,lossType)
    # saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save losses in a file
    # gt , pred = m.test(config,testData)
    # saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
    # adv_ae = getResults(gt,pred,shapeV,steps=config["smplBatchSize"])

    # # # ## Adversarial-SMPLAwareAutoEncoder
    # print(F"\n ~~~~~ Running ADV_SMPL_AE {lossType} {datasetName} ~~~~~~~~~~~ \n")
    # config["modelName"] = F"ADV_SMPL_AE_{datasetName}_{lossType}"
    # config["learningRate_des"] = 0.001
    # config["learningRate_gen"] = 0.004
    # m = AdversarialNetwork(genType="smpl_ae",inputType=inputType)
    # loss = m.train(config,trainData,lossType)
    # saveObjToFile(loss,F"{config['modelName']}_tl_vl_loss.json") # save losses in a file
    # gt , pred = m.test(config,testData)
    # saveObjToFile({"gt" : gt.tolist(),"pred":pred.tolist(),"shape":shapeV.tolist()},F"{config['modelName']}_gr_pred.json")
    # adv_smpl_ae = getResults(gt,pred,shapeV,steps=config["smplBatchSize"])



    return {
        F"r_2lr_{datasetName}_{lossType}_lb{lookBack}" : r_2lr,
        F"r_2lr_smpl_{datasetName}_{lossType}_lb{lookBack}" : r_2lr_smpl,
        F"r_adv_2lr_smpl{datasetName}_{lossType}_lb{lookBack}" : r_adv_2lr_smpl
        # F"r_vae_{datasetName}_{lossType}" : vae,
        # F"r_smpl_ae_{datasetName}_{lossType}" : smpl_ae,

        # F"r_adv_smpl_ae_{datasetName}_{lossType}":adv_smpl_ae,
        # F"r_adv_ae_{datasetName}_{lossType}":adv_ae
    }


def saveObjToFile(results,fileName) :
    fileLocation = os.path.join('results',fileName)
    if not os.path.isdir('results'):
        os.mkdir('results')

    try :
        with open(fileName,"w")  as fd :
            json.dump(results,fd)
    except Exception as e:
        print(F"unable to write to file {e}")

def printResult(fileName,nTests) :
    try :
        with open(fileName) as fd :
            data = json.load(fd)
    except Exception as e:
        print(F"unable to load file {fileName}")
        raise e
    
    results = {}
    for _d in data :
        for k in _d.keys() :
            for e in _d[k].keys() :
                k_name = F"{k}_{e}"
                if results.get(k_name,0) == 0:
                    results[k_name] = _d[k][e]
                else :
                    results[k_name] = results[k_name] + _d[k][e] # add them 
    
    # Average the results
    for k,v in results.items() :
        results[k] = v / nTests
    return results

if __name__ == "__main__" :

    config = {
        "learningRate" : 0.0001,
        "hiddenLayerSize" : 32,
        "numLayers" : 2,
        "batchSize" : 50,
        "epochs" : 500,
        "plotLoss" : False,
        "patience" : 20, # number of epochs to wait for Early Stopping
        "learningRateSchedular" : 5,
        "earlyStoppingDelta" : 0.000001,
        "modelName" : "2LR-LSTM.pth",
        "dataFileDir" : "data/pedx",
        "lookBack" : 5,
        "key" : "pose_orig_cam", # for which data model should run on
        "resultDir" : "results_behave",
        "smplBatchSize" : 200
    }

    NUM_TESTS = 3
    LOSS_TYPES = ["MSE","MAE"]
    DATASETS = ["PEDX","HYBRID","BEHAVE"]
    LOOK_BACKS = [5]
    results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(F"number of epochs {config['epochs']}, batchSize {config['batchSize']}")
    
    

    for nTest in range(1,NUM_TESTS) :

        for lookBack in LOOK_BACKS :
            config["lookBack"] = lookBack #running for multiple lookbacks

            for dataset in DATASETS[-1:] :
                #specify the file to be used based on dataset
                if dataset == "PEDX" :
                    config["dataFileDir"] = "data/pedx"
                elif dataset == "BEHAVE" :
                    config["dataFileDir"] = "data/behave"
                elif dataset == "ICSENS" :
                    config["dataFileDir"] = "data/icsens"
                elif dataset == "NTU" :
                    config["dataFileDir"] = "data/ntu"
                elif dataset == "HYBRID" :
                    config["dataFileDir"] = ["data/pedx","data/behave"]

                print(F"\n ##### Running all the models using dataset {dataset} with files in  {config['dataFileDir']} looback {lookBack}")
                mData = ModelData(config)
                
                trainData = mData.getTrainData()
                validData = mData.getValidData()
                testData = mData.getTestData()
                # with open("poseTestData.json",'w') as fd :
                #     json.dump({
                #         "x" : testData[0].tolist(),
                #         "y" : testData[1].tolist()
                #     },fd)

                lcTrainData = mData.getLcTrainData()
                lcValidData = mData.getLcValidData()
                lcTestData = mData.getLcTestData()
                # with open("frameDiffTestData.json",'w') as fd :
                #     json.dump({
                #         "x" : lcTestData[0].tolist(),
                #         "y" : lcTestData[1].tolist(),
                #         "t_1" : lcTestData[2].tolist()
                #     },fd)

                for lossType in LOSS_TYPES :

                    print(F"\n ##### lossType {lossType} , dataset {dataset} and lookback {lookBack}")

                    print(F"\n ####### Running Models with pose SMPL data with lossType : {lossType}")
                    print(F"Train data shape is {trainData[0].shape} , {trainData[1].shape}")
                    print(F"Validation data is in shape {validData[0].shape} ,{validData[1].shape}")
                    print(F"testing data is in shape {testData[0].shape},{testData[1].shape} \n \n")
                    
                    _out = runModels(config,trainData,validData,testData,lossType,dataset,'pose',lookBack,device,nTest) # run basic models
                    results.append(_out)

                    print(F"\n \n ###### Running Frame difference models with lossType : {lossType}")
                    print(F"Train data shape is {lcTrainData[0].shape} , {lcTrainData[1].shape}")
                    print(F"Validation data is in shape {lcValidData[0].shape} ,{lcValidData[1].shape}")
                    print(F"testing data is in shape {lcTestData[0].shape},{lcTestData[1].shape} \n \n")

                    _out = runModels(config,lcTrainData,lcValidData,lcTestData,lossType,dataset,"frame_diff",lookBack,device,nTest) # run basic models
                    results.append(_out)
            del mData  
            gc.collect()
    saveObjToFile(results, F"test_{nTest}_results.json")
    # print(printResult("modelResults_test_1.json",NUM_TESTS))
    print("this thing is completed")

# /home/ICTDOMAIN/d20125529/SMPL-LSTM/venv/bin