"""
Author : Kunchala Anil
Date : Jan 2 2021
This script is used to render SMPL meshes from VIBE or any other meshes

We will get SMPL pose data from VIBE and generate vertex faces using CalciferZh/SMPL implemention and render it using pyrender

I borrowed code from following repositories 
    VIBE
    CalcifierZh/SMPL

Steps :
    1. Need Extract the data from VIBE. this is done using getSMPLFromVibe.py generated ouput will be saved into out.json
    2. Extract SMPL pose data
    3. Generate SMPL mesh using CalcifierZh
    4. Render them using SMPL
"""

from utils.renderer import *
from utils.smpl_np import *
import os,json
import cv2

fileName = "out.json"
imagesDir = "/home/anil/Documents/VIBE/srcFiles" #Directory where source images are stored
smplModelFile = "model.pkl"

def loadData(fileName) :
    ## Load data from file
    try :
        with open(fileName) as fd :
            data = json.load(fd)
    except Exception as e :
        print(F"unable to get data from {fileName} , Please make sure file exists ")
        raise

    print(F"total number of persons found in the file are {len(data)}")
    return data


def renderMesh(pose,betas,orig_cam,img,smplModelFile,color=[0,1,0],orig_img=True):
    """
    img -> cv2.imread() of image 
    """
    # get SMPL vertices based on SMPL pose parameters
    smpl = SMPLModel(smplModelFile)
    smpl.set_params(pose=pose,beta=betas,trans=None)

    img_shape = img.shape
    orig_height, orig_width = img_shape[:2]
    renderer = Renderer(resolution=(orig_width, orig_height))
    img_updated = renderer.render(
            img,
            smpl.verts,
            orig_cam,
            smpl.faces,
            color=color,
            orig_img=orig_img
        )
    return img_updated

# we dont need to create render object everytime while using locally
def renderMesh2(img,smplverts,orig_cam,smplfaces,renderer,color=[0,1,0],orig_img=True):
    img_updated = renderer.render(
            img,
            smplverts,
            orig_cam,
            smplfaces,
            color=color,
            orig_img=orig_img
        )
    return img_updated

def main():
    data = loadData(fileName)
    #take the first person available in data and generate meshes
    selected_id = 3
    data_c = data[selected_id]

    #we will use starting frame to render results
    starting_frame = int(data_c['frames'][0])

    image_path = os.path.join(imagesDir,F"{starting_frame:06d}.jpg")
    img = cv2.imread(image_path) 
    img_out = img

    smpl = SMPLModel(smplModelFile)
    # smpl.set_params(pose=pose,beta=betas,trans=None)

    img_shape = img.shape
    orig_height, orig_width = img_shape[:2]
    renderer = Renderer(resolution=(orig_width, orig_height))

    for frame_id in range(len(data_c['frames'])):
    
        # and fixed size parameters
        betas = data_c['betas'][0]

        # get pose data for each frame
        print(F"currently running for frame : {data_c['frames'][frame_id]}")
        pose = np.array(data_c['pose'][frame_id])
        orig_cam = np.array(data_c['orig_cam'][frame_id])

        # get SMPL Mesh vertices based on pose
        # smpl = SMPLModel(smplModelFile)
        smpl.set_params(pose=pose,beta=betas,trans=None)
        
        # render the Mesh 
            
        # if first frame we will use base image to overlay mesh
        # if not we will just use updated overlay image as input so consequetive images will be added to base image
        if frame_id == 0 :
            # img = cv2.imread(image_path)
            # img_out = renderMesh(pose,betas,orig_cam,img,smplModelFile)
            img_out = renderMesh2(img,smpl.verts,orig_cam,smpl.faces,renderer,color=[1,1,0],orig_img=True)
        else :
            # img_out = renderMesh(pose,betas,orig_cam,img_out,smplModelFile)
            img_out = renderMesh2(img_out,smpl.verts,orig_cam,smpl.faces,renderer)

        # img_shape = img.shape
        # orig_height, orig_width = img_shape[:2]
        # renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True)
        # img_updated = renderer.render(
        #         img,
        #         smpl.verts,
        #         orig_cam,
        #         smpl.faces
        #     )
    imS = cv2.resize(img_out, (int(orig_width/2), int(orig_height/2))) # Resize image
    cv2.imshow('img',imS)
    cv2.waitKey(0) 
    #closing all open windows  
    cv2.destroyAllWindows()

"""
used to generate images and respective meshes for diagrams
"""
def generateIndividualMeshs():
    person_id = 3
    n_frames = 8
    data = loadData(fileName)
    data_c = data[person_id]

    #we will use starting frame to render results
    starting_frame = int(data_c['frames'][0])

    image_path = os.path.join(imagesDir,F"{starting_frame:06d}.jpg")
    img = cv2.imread(image_path) 

    smpl = SMPLModel(smplModelFile)
    # smpl.set_params(pose=pose,beta=betas,trans=None)

    img_shape = img.shape
    orig_height, orig_width = img_shape[:2]
    renderer = Renderer(resolution=(orig_width, orig_height))


    for frame_id in range(n_frames):
    
        # and fixed size parameters
        betas = data_c['betas'][0]

        # get pose data for each frame
        print(F"currently running for frame : {data_c['frames'][frame_id]}")
        pose = np.array(data_c['pose'][frame_id])
        orig_cam = np.array(data_c['orig_cam'][frame_id])

        # get SMPL Mesh vertices based on pose
        # smpl = SMPLModel(smplModelFile)
        smpl.set_params(pose=pose,beta=betas,trans=None)

        image_path = os.path.join(imagesDir,F"{frame_id:06d}.jpg")
        img = cv2.imread(image_path)
        
        # render the Mesh 
        img_out = renderMesh2(img,smpl.verts,orig_cam,smpl.faces,renderer,orig_img=False)
        img = img[1132:1132+680, 1180:1180+680 ]
        img_out = img_out[1132:1132+680,1180:1180+680]

        cv2.imwrite(F"/home/anil/Desktop/out/{frame_id:06d}_org.jpg",img)
        cv2.imwrite(F"/home/anil/Desktop/out/{frame_id:06d}_mesh.jpg",img_out)
        

if __name__ == '__main__' :
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl' 
    main()
    #generateIndividualMeshs()




