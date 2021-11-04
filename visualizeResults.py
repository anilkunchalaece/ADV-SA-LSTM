"""
This script is used to generate SMPL meshes using ground truth and predicted SMPL parameters 
"""
import os
import json
import cv2
from viewMeshes import renderMesh
import numpy as np


def visualizeResults(srcDir,desDir,smplModelFile,gt_pred_fileName):
    # orig_cam = [0.09646371219391524, 0.13085605915400858, 5.8277099413193305, 0.557122390235591]
    betas = (np.random.rand(10) - 0.5) * 0.06
    
    print(F"processing for {gt_pred_fileName}")
    
    with open(gt_pred_fileName) as fd :
        data = json.load(fd)

    for i in range(500) :    
        gt = data["gt"][i]
        pred = data["pred"][i]

        # print(F"gt => {gt[-3:]}")
        # print(F"pred => {pred[-3:]}")
        # pred[:3] = gt[:3]
        print(F"Running for {i}")

        image_path = os.path.join(srcDir,os.listdir(srcDir)[0]) # take first frame
        # print(image_path)
        img = cv2.imread(image_path)
        img_out = img     
        img_out = renderMesh(np.array(gt[:-4]),betas,np.array(gt[-4:]),img_out,smplModelFile,color=[0,0,1],orig_img=False) # for final mesh use different color for rendering
        img_out = renderMesh(np.array(pred[:-4]),betas,np.array(pred[-4:]),img_out,smplModelFile,color=[1,0,0],orig_img=True) # for final mesh use different color for rendering
        

        orig_height, orig_width = img.shape[:2]
        

        # imS = cv2.resize(img_out, (int(orig_width/4), int(orig_height/4))) # Resize image
        # cv2.imshow('img',imS)
        cv2.imwrite(os.path.join(desDir,F"{i}.png"),img_out)
        # cv2.waitKey(0) 
        #closing all open windows
    # cv2.destroyAllWindows()

def makeVideoWithImages(srcDir):
    img2 = cv2.imread(os.path.join(srcDir,r'0.png'))
    height , width = img2.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # choose codec according to format needed
    video=cv2.VideoWriter(os.path.join(srcDir,'video.mp4'), fourcc, 1,(width,height))

    for j in range(0,120):
        img = cv2.imread(os.path.join(srcDir,str(j)+'.png'))
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

def main():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # srcDir = "/media/anil/Elements/behaveData/data" #Directory where source images are stored
    srcDir = "/home/anil/Documents/VIBE/srcFiles"# for pedx
    smplModelFile = "model.pkl"

    d="/home/anil/Documents/SMPL-LSTM/results/modelsWithResults/models/HYBRID"

    filesToProcess = [
                F"{d}/2LR_LSTM_HYBRID_MAE_frame_diff_lb5_t1_gr_pred.json",
                F"{d}/2LR_LSTM_HYBRID_MAE_pose_lb5_t1_gr_pred.json",
                F"{d}/2LR_LSTM_HYBRID_MSE_frame_diff_lb5_t1_gr_pred.json",
                F"{d}/2LR_LSTM_HYBRID_MSE_pose_lb5_t1_gr_pred.json",
                F"{d}/ADV_SMPL_AWARE_LR_LSTM_HYBRID_MAE_lb5_t1_gr_pred.json",
                F"{d}/SMPL_AWARE_LR_LSTM_HYBRID_MAE_frame_diff_lb5_t1_gr_pred.json"
            ]

    for fName in filesToProcess :

        desDir = os.path.basename(fName).replace("_lb5_t1_gr_pred.json","")
        desDir = "/home/anil/Documents/SMPL-LSTM/results/modelsWithResults/visualizations/hybrid/" + desDir
        if not os.path.isdir(desDir):
            os.mkdir(desDir)
        visualizeResults(srcDir,desDir,smplModelFile,fName)

def getCropedImage(imgPath):
    imgOrg = cv2.imread(imgPath)
    img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    row,col = img.shape
    rows = []
    cols = []
    # for i in range(row):
    #     for j in range(col) :
    #         if img[i,j] != 1 :
    #             rows.append(i)
    #             cols.append(j)
    c = np.argwhere(img!=1)
    rmin , rmax = np.min(c[:,0]) , np.max(c[:,0])
    cmin , cmax = np.min(c[:,1]) , np.max(c[:,1])
    crop = imgOrg[rmin:rmax, cmin:cmax]
    # crop = np.dstack((crop, alpha_channel))
    # print(crop.shape)
    # cv2.imwrite("out.png",crop)
    return crop

# define a function for horizontally 
# concatenating images of different
# heights 
def hconcat_resize(img_list, 
                   interpolation 
                   = cv2.INTER_CUBIC):
    imgList = []
    black = [0,0,0]
    for img in img_list :
        imgList.append(cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=black))
    # print(imgList)
    # take minimum hights
    h_min = min(img.shape[0] 
                for img in imgList)

    
      
    # image resizing 
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in imgList]
      
    # return final image
    return cv2.hconcat(im_list_resize)

def renderAllOutputs() : 
    # imgPath = "./pedx/2LR_LSTM_PEDX_MAE_pose/9.png"
    dirName = "/home/anil/Documents/SMPL-LSTM/results/modelsWithResults/visualizations/hybrid"
    outDir = "/home/anil/Documents/SMPL-LSTM/results/modelsWithResults/visualizations/hybridOut"
    
    for i in range(500) :
        imgList = []
        for d in sorted(os.listdir(dirName)) :
            # print(d)
            _out = getCropedImage(os.path.join(dirName,d,F"{i}.png"))
            imgList.append(_out)

        hImg = hconcat_resize(imgList)
        outImgPath = os.path.join(outDir,F"{i}.png")
        cv2.imwrite(outImgPath,hImg)
        print(F"complted {i}")

if __name__ == "__main__" :
    # main() # used to generate gt and pred visualizations using *_gt_pred.json
    # makeVideoWithImages("Gt_Pred_without_root")
    renderAllOutputs()
