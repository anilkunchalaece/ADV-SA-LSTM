"""
This script is used to generate SMPL-LSTM results for given directory
    - Preprocess images in directory for VIBE [preprocess_pedx.py]
    - Run VIBE for images [demo_modified.py]
    - Extract SMPL data from VIBE [getSMPLfromVibe.py]- takes pkl file and convert it to json [getSMPLFromVibe.py]
"""

import os
from posixpath import dirname
import preprocess_pedx
import getSMPLFromVibe

# processed output will be stored in tmpDir/out.json
def main(srcDir,tmpDir):
    dirList = getDirList(srcDir)
    vibeOutDir = os.path.join(tmpDir,'out') # dir to store vibeOut

    for idx,dirName in enumerate(dirList) : 
        print(F"running for {dirName}")
        renameImages(dirName)
        #preprocess images for vibe
        processedDir = os.path.join(vibeOutDir,dirName.split("/")[-2],os.path.basename(dirName)) # idx should be used for only pedx since all dir has same name
        print(F"Preprocessing images in {dirName} and saving them in {processedDir}")
        # preprocess_pedx.processImages(dirName,processedDir,steps=300) # create subdirectories with images

        print(F"Running VIBE for images in {processedDir}")
        # since I'm unable to run vibe for all images , we need to run for each subdir in processedDir
        # for _d in os.listdir(processedDir) :
        #     _d_path = os.path.join(processedDir,_d)
        #     _vibe_out = os.path.join(vibeOutDir,F"_{idx}_{_d}")
        if os.path.isdir(processedDir) :
            #print(F"{_vibe_out} is processed , skipping")
            pass
        else :
            print(F"{processedDir} not processed , running now")
            runVibeOnImages(dirName,processedDir)

    # Once VIBE is completed , extract all SMPL data into single file
    vibeOutList = [ os.path.join(vibeOutDir,_d,'left','ex') for _d in os.listdir(vibeOutDir)]
    print(vibeOutList)
    getSMPLFromVibe.extractSMPLData(vibeOutList,os.path.join(tmpDir,'icSens'))
        

def runVibeOnImages(srcDir,outDir) :
    cmdForVibe = F"python demo_modified.py --images_dir {srcDir} --output_folder {outDir}"# --no_render"
    
    vibeSrcDir = "/home/anil/Documents/VIBE"
    # vibeSrcDir = "/home/ICTDOMAIN/d20125529/VIBE"
    vibeVENV = os.path.join(vibeSrcDir,'vibe-env')

    # Activate venv, run VIBE , deactivate venv
    cmdToRun = F"cd {vibeVENV} && . bin/activate && cd {vibeSrcDir} && {cmdForVibe} && cd {vibeVENV} && . dectivate"
    os.system(cmdToRun)

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def renameImages(dirName) :
    import shutil
    print(dirName)
    filesList = sorted(os.listdir(dirName),reverse=False)
    print(filesList[0])
    startingValue = int(filesList[0].split(".")[0][-3:])
    for fileName in os.listdir(dirName) :
        _new_name = int(fileName.split("_")[-1].split(".")[0][-3:]) #- startingValue
        _new_name = F"{_new_name:06d}{fileName[-4:]}"
        src = os.path.join(dirName,fileName)
        des = os.path.join(dirName,_new_name)
        shutil.move(src,des)
    

 
def getDirList(dirName) :
    # pedxDataDir = "/media/anil/Elements/pedxData/data"
    # consider only single camera images in pedx dataset
    dirList = []
    # dirName = "/media/anil/Elements/icSenseData/data/view1"
    for _d in os.listdir(dirName) :
        # for _dd in os.listdir(os.path.join(dirName,_d)) :
        #     dirList.append(os.path.join(dirName,_d,_dd))
        dirList.append(os.path.join(dirName,_d,'left'))

    return dirList

if __name__ == "__main__" :
    # vibeSrcDir = "/home/anil/Documents/VIBE"
    DIR_NAME = "/media/anil/Elements/icSenseData/data"
    for _d in os.listdir(DIR_NAME):
        srcDir = os.path.join(DIR_NAME,_d)
        tmpDir = os.path.join("/media/anil/Elements/tmp/icSens",_d) #Temp dir to store all the processed images
        main(srcDir,tmpDir)
        print(F"completed for {_d}")
    # print(getDirList())














