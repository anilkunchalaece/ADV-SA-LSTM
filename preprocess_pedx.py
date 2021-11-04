"""
Author : Kunchala Anil
Date : December 15 2020

This script is used to change the files names in data set / image dir to %06d format
since VIBE by default accepts that format , currently I'm not planning to change
VIBE source code, so need to go by its rules

"""

import os
import shutil
import argparse

# srcDir = "/home/anil/Downloads/20171130T2000/blu79CF"
# desDir = "/home/anil/Documents/VIBE/src2" # dir to write processed files

def processImages(srcDir,desDir,steps=None):
    filesList = sorted(os.listdir(srcDir),reverse=False)
    dirName = 'behave'
    if dirName == "pedx" :
        startingValue = int(filesList[0].split("_")[-1].split(".")[0])
    else :
        print(filesList[0])
        startingValue = int(filesList[0].split(".")[0])
    
    org_des = desDir
    stepIdx = 1
    
    # #create desDir if not exists
    # if os.path.exists(desDir) == False :
    #     os.makedirs(desDir,exist_ok=True)
    
    for idx,fileName in enumerate(filesList) :
        # VIBE unable to run for total images , so we are splitting the images into sets to run
        # I suck. please bear with me :(
        # print(idx,idx%steps==0)
        if steps != None and idx % steps == 0 :
            desDir = os.path.join(org_des,str(stepIdx))
            if os.path.exists(desDir) == False :
                os.makedirs(desDir)
                # if idx != 0 : #ignore first frame
                    # startingValue = idx + 1 # reset the starring value
                # startingValue = int(fileName.split("_")[-1].split(".")[0])
                stepIdx = stepIdx + 1 # create another directory to store images
                # desDir = os.path.join(org_des,str(stepIdx))                
        
        _new_name = int(fileName.split("_")[-1].split(".")[0]) #- startingValue
        _new_name = F"{_new_name:06d}{fileName[-4:]}"
        src = os.path.join(srcDir,fileName)
        des = os.path.join(desDir,_new_name)
        shutil.copy2(src,des)
        
        if idx % 500 == 0 : # print once for 200 images
            print(F"copied {src} to {des}")
        

    print("processing is completed")    


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", help="Directory of Images to be processed")
    parser.add_argument("--desDir",help="Directory to stored renamed images")
    
    args = parser.parse_args()
    srcDir = args.srcDir
    desDir = args.desDir

    processImages(srcDir,desDir)
