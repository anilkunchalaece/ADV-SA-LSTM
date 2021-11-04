"""
Author : Kunchala Anil
This is used to preprocess data suitable for LSTM
It is used to split data based on the lookback window 

Inspired from these :
https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

Need to write this more optimally 

"""

import json
import sys
import numpy as np
import cv2
import os
import random

from numpy.lib.function_base import select
from viewMeshes import renderMesh
from concurrent.futures import ThreadPoolExecutor

def loadData(dataFile,lookback) :
    data = getDataFromFile(dataFile)

    poseData = []
    frameData = []
    betasData = []
    orig_camData = []
    joints_3d = []
    trans = []
    poseAndTrans = []
    poseAndCam = []

    total = 0
    for v in data :
        totalFrames = len(v['frames'])
        # print(totalFrames)
        for i in range(lookback,totalFrames) :
            total = total + 1
            _pose = []
            _frames = []
            _betas = []
            _orig_cam = []
            _joints_3d = []
            _pose_and_trans =[]
            _pose_and_orig_cam = []

            # Split data based on lookback window
            _pose.extend(v['pose'][i-lookback : i+1]) 
            _frames.extend(v['frames'][i-lookback : i+1])
            _betas.extend(v['betas'][i-lookback : i+1])
            _orig_cam.extend(v['orig_cam'][i-lookback : i+1])
            _joints_3d.extend(v['joints3d'][i-lookback : i+1])
            _pose_and_trans.extend(v['pose_and_trans'][i-lookback : i+1])
            _pose_and_orig_cam.extend(v['pose_orig_cam'][i-lookback : i+1])  


            poseData.append(_pose)
            frameData.append(_frames)
            betasData.append(_betas)
            orig_camData.append(_orig_cam)
            joints_3d.append(_joints_3d)
            poseAndTrans.append(_pose_and_trans)
            poseAndCam.append(_pose_and_orig_cam)

    #reshapings joints 3d
    # converting 3,49 joints to single dimension 3*49, so we can easily load into model 
    joints_3d = np.array(joints_3d)
    joints_3d = joints_3d.reshape(joints_3d.shape[0],joints_3d.shape[1],joints_3d.shape[2]*joints_3d.shape[3])

    return {
        "pose" : np.array(poseData),
        "frames" : np.array(frameData),
        "betas" : np.array(betasData),
        "orig_cam" : np.array(orig_camData),
        "joints_3d" : joints_3d,
        "pose_and_trans" : np.array(poseAndTrans),
        "pose_and_orig_cam" : np.array(poseAndCam)
    }

def loadData2(dataFolder,lookback,key) :
    """
    instead of loading entire file into memory - large file is split in to small ones where each contains data per persons
    - frames , pose , camera , orig_camera parameters
    """
    # data = getDataFromFile(dataFile)
    rData = []
    if type(dataFolder) != list :
        for fileName in os.listdir(dataFolder) :
            filePath = os.path.join(dataFolder,fileName)
            data = json.load(open(filePath))
            totalFrames = len(data['frames'])
            # print(totalFrames)
            for i in range(lookback,totalFrames) :
                _data = []
                # Split data based on lookback window
                _data.extend(data[key][i-lookback : i+1])
                rData.append(_data)
    else :
        nFilesToRead = min([len(os.listdir(x)) for x in dataFolder])
        # print(type(os.listdir(dataFolder[-1])))
        behaveFilesList = random.sample(os.listdir(dataFolder[-1]),nFilesToRead)

        for fileName in os.listdir(dataFolder[0]) :
            filePath = os.path.join(dataFolder[0],fileName)
            data = json.load(open(filePath))
            totalFrames = len(data['frames'])
            # print(totalFrames)
            for i in range(lookback,totalFrames) :
                _data = []
                # Split data based on lookback window
                _data.extend(data[key][i-lookback : i+1])
                rData.append(_data)

        for fileName in behaveFilesList :
            filePath = os.path.join(dataFolder[-1],fileName)
            data = json.load(open(filePath))
            totalFrames = len(data['frames'])
            # print(totalFrames)
            for i in range(lookback,totalFrames) :
                _data = []
                # Split data based on lookback window
                _data.extend(data[key][i-lookback : i+1])
                rData.append(_data)
    return np.array(rData)

def getDataFromFile(dataFile) :
    try :
        with open(dataFile) as fd :
            return json.load(fd)
    except FileNotFoundError :
        print("file not exists , please check the --data_file argument")
        raise
    except Exception as e :
        print(F"Some other error occured : {e}" )
        raise

def visualizeData(dataFile,look_back,srcDir,smplModelFile):
    data = loadData(dataFile,look_back)
    print(len(data['frames']))
    print(F"total number of samples are {len(data['pose'])}")
    
    data_length = look_back + 1

    for idx in range(len(data["frames"])):
        #load the first sample
        # selected_index = 25
        selected_index = idx

        frames = data['frames'][selected_index]
        pose = data['pose'][selected_index]
        betas = data['betas'][selected_index]
        orig_cam = data['orig_cam'][selected_index]

        image_path = os.path.join(srcDir,os.listdir(srcDir)[0]) # take first frame
        img = cv2.imread(image_path)
        img_out = img      
        
        for i in range(0,data_length) :
            if i == 0 :
                img_out = renderMesh(np.array(pose[i]),betas[i],orig_cam[i],img,smplModelFile)
            else :
                if i == (data_length - 1) :
                    img_out = renderMesh(np.array(pose[i]),betas[i],orig_cam[i],img_out,smplModelFile,color=[0,0,1],orig_img=True) # for final mesh use different color for rendering
                else :
                    img_out = renderMesh(np.array(pose[i]),betas[i],orig_cam[i],img_out,smplModelFile)

        orig_height, orig_width = img.shape[:2]             

        imS = cv2.resize(img_out, (int(orig_width/2), int(orig_height/2))) # Resize image
        # cv2.imshow('img',imS)
        # cv2.waitKey(0)
        print(idx) 
        cv2.imwrite(F"out/{idx}.png",imS)
    #closing all open windows  
    cv2.destroyAllWindows()

def visualizeResults(srcDir,smplModelFile):
    orig_cam = [0.09646371219391524, 0.13085605915400858, 5.8277099413193305, 0.557122390235591]
    betas = (np.random.rand(10) - 0.5) * 0.06

    with open("ADV_SMPL_AWARE_LR_LSTM_BEHAVE_MAE_lb5_t_2_gr_pred.json") as fd :
        data = json.load(fd)

    for i in range(10,100) :    
        gt = data["gt"][i]
        pred = data["pred"][i]

        print(F"gt => {gt[:3]}")
        print(F"pred => {pred[:3]}")
        pred[:3] = gt[:3]

        image_path = os.path.join(srcDir,F"{0:06d}.jpg") # take first frame
        img = cv2.imread(image_path)
        img_out = img     
        img_out = renderMesh(np.array(gt[:-4]),betas,np.array(gt[-4:]),img_out,smplModelFile,color=[0,0,1]) # for final mesh use different color for rendering
        img_out = renderMesh(np.array(pred[:-4]),betas,np.array(pred[-4:]),img_out,smplModelFile,color=[0,1,1]) # for final mesh use different color for rendering
        orig_height, orig_width = img.shape[:2]             

        imS = cv2.resize(img_out, (int(orig_width/4), int(orig_height/4))) # Resize image
        cv2.imshow('img',imS)
        cv2.waitKey(0) 
        #closing all open windows  
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl' 

    dataFile = "data/modified_pedx_out.json"
    # dataFile = "/media/anil/Elements/tmp/out.json"
    look_back = 4
    srcDir = "/home/anil/Documents/VIBE/srcFiles" #Directory where source images are stored
    smplModelFile = "model.pkl" 
    visualizeData(dataFile,look_back,srcDir,smplModelFile) # uncomment this line to view meshes
    # data = loadData(dataFile,look_back)
    # for k in data :
    #    print(F"{k} - {np.array(data[k]).shape}") 
    #visualizeResults(srcDir,smplModelFile)
