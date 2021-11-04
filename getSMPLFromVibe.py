"""
Author : Kunchala Anil
Date : 15 Dec 2020
This script is used to get SMPL parameters from VIBE output .pkl file and save it to JSON so we can use them in algorithm
"""
import os
import joblib
import json
import numpy as np

SRC_FILE = "vibe_output.pkl"

def getConsecutiveMaxArray(arr): 
    indexes = []
    final = []
    end = 0
    start = 0
    for i in range(1,len(arr)) :
        if arr[i] - arr[i-1] == 1 :
            end = i
        else :
            # print(start,end)
            final.append(arr[start:end+1])
            indexes.append((start,end))
            start = i
        if i == len(arr) - 1 :
            final.append(arr[start:end+1])
            indexes.append((start,end))
    max_lst = max(final,key=len)
    max_lst_i = final.index(max_lst)
    max_indexes = indexes[max_lst_i]
    return max_lst,max_indexes

# ref - https://github.com/mkocabas/VIBE/issues/172
def getTranslationParameters(orig_cam):
    img_size = 1280 # TODO need to check this
    flength = 500
    cam_s = float(orig_cam[0:1][0])
    cam_pos = orig_cam[2:]
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    return trans 


def extractData(data) :

    out_data = []

    for k in data :
        print(F" person id : {k} is in {data[k]['frame_ids'].shape} frames ")
        frame_ids = data[k]['frame_ids'].tolist()
        max_conse_lst , max_lst_indexes = getConsecutiveMaxArray(frame_ids)
        pose = data[k]['pose'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        verts = data[k]['verts'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        betas = data[k]['betas'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        pred_cam = data[k]['pred_cam'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        orig_cam = data[k]['orig_cam'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        bboxes = data[k]['bboxes'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        joints3d = data[k]['joints3d'].tolist()[max_lst_indexes[0]:max_lst_indexes[1]+1]
        pose_and_trans = [ list(np.hstack([pose[i] ,getTranslationParameters(orig_cam[i])])) for i in range(len(pose))] # append pose data to translation data
        pose_and_orig_cam = [ list(np.hstack([pose[i] ,orig_cam[i]])) for i in range(len(pose))] # append pose data to orig_cam
        
        out_data.append({
            "person_id" : k,
            "frames" : max_conse_lst,
            "pred_cam" : pred_cam,# translation parameters
            "orig_cam" : orig_cam,
            "pose" : pose,
            "betas" : betas,
            "bboxes" : bboxes,
            "joints3d" : joints3d,
            "pose_and_trans" : pose_and_trans,
            "pose_orig_cam" : pose_and_orig_cam
            
        })
    return out_data




def extractSMPLData(dirList,outFileName):
    out = []
    for des in dirList :
        try :
            print(F"Extracting data from {des} ")
            src = os.path.join(des,SRC_FILE) # write seperate file for each person/category
            fd = open(src,'rb')
            data = joblib.load(fd)
            out.extend(extractData(data))
        except Exception as e :
            print(F"unable to extract data from {des} : {e}")
    
    try :
        for i ,_data in enumerate(out) :
            # outFileName = F"{outFileName}{i}.json"
            with open(F"{outFileName}_{i}.json","w") as fd :
                json.dump(_data,fd)
    except Exception as e :
        raise Exception(F"unable to write data to {outFileName} : {e}")
    
    print(F"Data extraction completed, data is written to {outFileName}")

    # return out    

if __name__ == "__main__" :
    dirList = ["/home/anil/Documents/VIBE/output/ex" , "/home/anil/Documents/VIBE/output2/ex2"]
    outFileName = "out.json" # where to store data
    
    extractSMPLData(dirList,outFileName)