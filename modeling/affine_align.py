import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

'''
Affine align operation GPU version. based on torch.nn.functional.affine_grid() Using bilinear
Usage:
    -> features: (bz, 3, H, W) | (bz, 1, H, W) Variable
    -> idxs: (N, ) numpy array
    -> align_size: the size of output. 
    -> Hs: result of calcAffineMatrix(). a (N, 2, 3) size numpy array 
    
    <- rois: feature rois after align. (N, 3, align_size, align_size) | (N, 1, align_size, align_size) Variable
Example:
    NN, _, H_feat, W_feat = features.shape[0:4]
    Hs = calcAffineMatrix(H_feat, W_feat, keypoints, align_size, template_size, Haug)
    features_roi = affine_align_gpu(features_var, indexs, align_size, Hs)
'''
def affine_align_gpu(features, idxs, align_size, Hs):
    
    def _transform_matrix(Hs, w, h):
        _Hs = np.zeros(Hs.shape, dtype = np.float32)
        for i, H in enumerate(Hs):
            try:
                H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
                A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
                A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h/ 2.0], [0, 0, 1]])
                H0 = A.dot(H0).dot(A_inv)
                H0 = np.linalg.inv(H0)
                _Hs[i] = H0[:-1]
            except:
                print ('[error in (affine_align_gpu)]', H)
        return _Hs
    
    bz, C_feat, H_feat, W_feat = features.size()
    N = len(idxs)
    feature_select = features[idxs] # (N, feature_channel, feature_size, feature_size)
    # transform coordinate system
    Hs_new = _transform_matrix(Hs[:, 0:2, :], w=W_feat, h=H_feat) # return (N, 2, 3)
    Hs_var = Variable(torch.from_numpy(Hs_new).cuda(), requires_grad=False)
    ## theta (Variable) – input batch of affine matrices (N x 2 x 3)
    ## size (torch.Size) – the target output image size (N x C x H x W) 
    ## output Tensor of size (N x H x W x 2)
    flow = F.affine_grid(theta=Hs_var, size=(N, C_feat, H_feat, W_feat)).float().cuda()
    flow = flow[:,:align_size[0], :align_size[1], :]
    ## input (Variable) – input batch of images (N x C x IH x IW)
    ## grid (Variable) – flow-field of size (N x OH x OW x 2)
    ## padding_mode (str) – padding mode for outside grid values ‘zeros’ | ‘border’. Default: ‘zeros’
    rois = F.grid_sample(feature_select, flow, mode='bilinear', padding_mode='zeros') # 'zeros' | 'border' 
    return rois

def visualizeOutput(self, netOutput):
    outdir = './vis/'
    print("entered function")
    netOutput = netOutput.transpose(0, 2, 3, 1)
    MaskOutput = [[] for _ in range(self.bz)]

    mVis = translib.stride_matrix(4)

    idx = 0
    for i, (img, masks) in enumerate(zip(self.batchimgs, self.batchmasks)):
        height, width = img.shape[0:2]
        for j in range(len(masks)):
            predmap = netOutput[idx]

            predmap = predmap[:, :, 1]
            predmap[predmap>0.5] = 1
            predmap[predmap<=0.5] = 0
            predmap = cv2.cvtColor(predmap, cv2.COLOR_GRAY2BGR)
            predmapint = predmap
            predmap = cv2.warpAffine(predmap, mVis[0:2], (256, 256))

            matrix = self.maskAlignMatrixs[i][j]
            matrix = mVis.dot(matrix)
            print("H : ", matrix)
            invmatrix = np.linalg.inv(matrix)
            print("invmatrix: ",invmatrix)
            # invmatrix = invmatrix/invmatrix[-1,-1]
            imgRoi = cv2.warpAffine(img, matrix[0:2], (256, 256))

            mask = cv2.warpAffine(masks[j], matrix[0:2], (256, 256))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


            I = np.logical_and(mask, predmap)
            U = np.logical_or(mask, predmap)
            iou = I.sum() / U.sum()

            predmap2 = cv2.warpAffine(predmap,matrix[0:2],(width, height),flags= cv2.WARP_INVERSE_MAP)

            vis = np.hstack((imgRoi, mask*255, predmap*imgRoi))
            print(os.path.abspath('./'))
            print("writing to: ",outdir + '%d_%d_%.2f.jpg'%(self.visCount, j, iou))
            cv2.imwrite(outdir + '%d_%d_%.2f.jpg'%(self.visCount, j, iou), np.uint8(vis))
            cv2.imwrite(outdir + '%d_%d_%.2f_invmask.jpg'%(self.visCount, j, iou), np.uint8(predmap2*255))
            cv2.imwrite(outdir + '%d_%d_%.2f_inv.jpg'%(self.visCount, j, iou), np.uint8(predmap2*img))
            idx += 1