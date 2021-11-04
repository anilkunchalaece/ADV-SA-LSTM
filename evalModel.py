"""
Author : Kunchala Anil
This script has helper functions to evalutate the Model.
"""
import numpy as np
import cv2
import json
from torchModels import *

def alignByRoot(joints):
    """
    Assumes joints is 24 x 3 in SMPL order.
    Subtracts the location of the root joint from all the other joints
    """
    root = joints[0, :]

    return joints - root

def MPJAE(pred_rotation, gt_rotation):
    """
    Computes Mean Per Joint Angle Error for SMPL 24 Joint rotations
    Params :
        pred_rotations - Predicted SMPL rotations in rotation matrix representations of shape [24,3,3]
        gt_rotations - Ground Truth SMPL rotations in rerotation matrix representation of shape [24,3,3]
    """
    pred_rotation_t = np.transpose(pred_rotation,[0, 2, 1])
    # compute pred_rotation.T * gt_rotation, if prediction and target match, this will be the identity matrix
    r = np.matmul(gt_rotation,pred_rotation_t)
    angles = []

    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r.shape[0]) :
        aa, _ = cv2.Rodrigues(r[i,:,:])
        angles.append(np.linalg.norm(aa)) # get rotation from axis angle representations 
    
    return np.mean(angles)

def MPJAEWithoutRoot(pred_rotation, gt_rotation):
    """
    Computes Mean Per Joint Angle Error for SMPL 24 Joint rotations
    Params :
        pred_rotations - Predicted SMPL rotations in rotation matrix representations of shape [24,3,3]
        gt_rotations - Ground Truth SMPL rotations in rerotation matrix representation of shape [24,3,3]
    """
    # remove the first angle , i.e root rotation
    pred_rotation = pred_rotation[1:,:,:]
    gt_rotation = gt_rotation[1:,:,:]

    pred_rotation_t = np.transpose(pred_rotation,[0, 2, 1])
    # compute pred_rotation.T * gt_rotation, if prediction and target match, this will be the identity matrix
    r = np.matmul(gt_rotation,pred_rotation_t)
    angles = []

    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r.shape[0]) :
        aa, _ = cv2.Rodrigues(r[i,:,:])
        angles.append(np.linalg.norm(aa)) # get rotation from axis angle representations 
    
    return np.mean(angles)
    

def MPJPE(pred_joints_3d, gt_joints_3d):
    """
    Computes Mean Per Joint Error for SMPL 24 Joints 
    Params : 
        pred_joints_3d - Predicted SMPL 3D joints of shape [24,3]
        gt_joints_3d - Ground Truth SMPL 3D Joints  of shape [24,3]

    returns :
        MPJPE in mm
    """
    # print(gt_joints_3d.shape)
    gt_joints_3d = alignByRoot(gt_joints_3d)
    pred_joints_3d = alignByRoot(pred_joints_3d)

    joint_error = np.sqrt(np.sum((gt_joints_3d - pred_joints_3d) ** 2, axis=1))
    return np.mean(joint_error)

def vertexRMSE(pred_v,gt_v):
    """
    Calculate Vertex RMSE error
    params :
        pred_v - Predicted SMPL vertex of shape [6890,3]
        gt_v - Ground Truth SMPL vetex of shape [6890,3]

    returns :
        vertexRMSE in mm
    """
    v_rmse = np.sqrt(np.mean(np.sqrt(np.sum((pred_v - gt_v)**2,axis=1))))
    return v_rmse

def getTransError(pred,gt) :
    return np.sqrt(np.mean(np.sqrt(np.sum((pred[:,-4:-2] - gt[:,-4:-2])**2,axis=1)))) * 1000

def getGlobalRotError(pred,gt) :
    # Calculate the global / root rotation error
    gr_e = [] # global rotation error
    for x in range(pred.shape[0]) :
        gt_r = cv2.Rodrigues(gt[x,:3])[0] # convert axis angle to rotation matrix
        pred_r = cv2.Rodrigues(pred[x,:3])[0] # convert axis angle to rotation matrix
        pred_r_t = np.transpose(pred_r) # transpose / inverse matrix of pred_r
        # print(gt_r[0].shape,pred_r.shape)
        r = np.matmul(gt_r,pred_r_t)
        a,_ = cv2.Rodrigues(r)
        gr_e.append(np.linalg.norm(a))
    gr_e_final = np.degrees(np.mean(np.array(gr_e)))
    return gr_e_final


def getTransErrorOrg(pred,gt) :
    
    # Calculate the translation error

    # tr_e =  np.mean(np.sqrt((gt[:,-4:-2]-pred[:,-4:-2])**2))
    tr_e = np.sqrt(np.mean(np.sqrt(np.sum((pred[:,-4:-2] - gt[:,-4:-2])**2,axis=1))))

    # Calculate the global / root rotation error
    gr_e = [] # global rotation error
    for x in range(pred.shape[0]) :
        gt_r = cv2.Rodrigues(gt[x,:3])[0] # convert axis angle to rotation matrix
        pred_r = cv2.Rodrigues(pred[x,:3])[0] # convert axis angle to rotation matrix
        pred_r_t = np.transpose(pred_r) # transpose / inverse matrix of pred_r
        # print(gt_r[0].shape,pred_r.shape)
        r = np.matmul(gt_r,pred_r_t)
        a,_ = cv2.Rodrigues(r)
        gr_e.append(np.linalg.norm(a))
    gr_e_final = np.degrees(np.mean(np.array(gr_e)))
    
    return gr_e_final,tr_e*1000


if __name__ == "__main__" :
    import statistics
    d = "/home/anil/Documents/SMPL-LSTM/results/modelsWithResults/models/PEDX/"
    shapeV = ((np.random.rand(1,10) - 0.5)*0.06) # random shape value used for all models for testing

    v_rmse = []
    mpjae = []
    mpjpe = []
    mpjae_without_root = []
    gr_e = []
    tr_e = []
    out = {}
    for t in range(3) :

        fNames = [                    
                    F"{d}2LR_LSTM_PEDX_MSE_pose_lb5_t{t}_gr_pred.json",
                    F"{d}2LR_LSTM_PEDX_MSE_frame_diff_lb5_t{t}_gr_pred.json",
                    F"{d}2LR_LSTM_PEDX_MAE_pose_lb5_t{t}_gr_pred.json",
                    F"{d}2LR_LSTM_PEDX_MAE_frame_diff_lb5_t{t}_gr_pred.json",
                    F"{d}SMPL_AWARE_LR_LSTM_PEDX_MAE_frame_diff_lb5_t{t}_gr_pred.json",
                    F"{d}ADV_SMPL_AWARE_LR_LSTM_PEDX_MAE_lb5_t{t}_gr_pred.json"
                    ]
        for f in fNames : 
            print(f)
            with open(f) as fd :
                data = json.load(fd)
            gt = np.array(data["gt"])
            pred = np.array(data["pred"])
            r = getResults(gt,pred,shapeV,None,steps=200)
            k = "_".join(os.path.basename(f).split("_")[:5])
            if out.get(k,1) == 1 :
                out[k] = []
            out[k].append(r)
    for k,v in out.items() :
        keys_ = ["vertex","mpjpe","mpjae","mpjae_without_root","gr_error","tr_error"]
        print(k)
        for k_ in keys_ :
            v_out = [x[k_] for x in v]
            print(F"{k_},{statistics.mean(v_out):0.2f}({statistics.stdev(v_out):0.2f})",end=" ")
        print("")
    