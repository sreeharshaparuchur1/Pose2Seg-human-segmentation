import sys
sys.path.insert(0, '../')

import random
import os
import numpy as np
import os.path as osp
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

from lib.timer import Timers
from modeling.resnet import resnet50FPN
from modeling.seg_module import resnet10units
import lib.transforms as translib
from modeling.skeleton_feat import genSkeletons
from matplotlib import pyplot as plt
from modeling.affine_align import affine_align_gpu
from modeling.core import PoseAlign

timers = Timers()

def plot_figure(img_arr,a,b,text='',c=1):
    name = "./r_img/55-skeleton-features-" + text + ".jpg"
    final_arr = np.sum(img_arr,axis = 0)
    figx = plt.figure()
    xxx = figx.add_subplot(1,1,1)
    xx = xxx.imshow(final_arr,cmap="viridis")
    figx.colorbar(xx, ax = xxx)
    xname = "./r_img/combined-skeleton-features-" + text + ".jpg"
    plt.savefig(xname)
    figx.clf()


    img_arr = np.abs(img_arr)
    fig = plt.figure(figsize=(b*5,a*5+1))
    for i in range(1,a+1):
        for j in range(1,b+1):
            ax=fig.add_subplot(a,b,(i-1)*b+j)
            if c==3:
                ax.imshow(np.abs(img_arr[(i-1)*b+j-1]))
            if c==1:
                ax.imshow(np.abs(img_arr[(i-1)*b+j-1]),cmap='gray')
            ax.axis("off")
            ax.set_title("image - "+str((i-1)*b+j)+ " " +str(np.max(img_arr[(i-1)*b+j-1])) + " " + str(np.min(img_arr[(i-1)*b+j-1])), fontsize=12)
    
    plt.suptitle(text, fontsize=15)
    #plt.tight_layout()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,rect=[0, 0, 1, 0.95])
    #fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    # plt.show()
    # name = "./r_img/sk_fts-" + text + ".jpg"
    # plt.savefig(name)
    # final_arr = np.sum(img_arr,axis = 0)
    # figx = plt.figure()
    # xxx = figx.add_subplot(1,1,1)
    # xxx.imshow(final_arr,cmap="viridis")
    # xname = "./r_img/x-" + text + ".jpg"
    # plt.savefig(xname)
    plt.savefig(name)
    fig.clf()
    return 

def plot_figure2(img_arr,a=16,b=16,text='',c=3):
    name = "./r_img/256-resnet50-features-" + text + ".jpg"
    
    fig = plt.figure(figsize=(b*5,a*5+1))
    for i in range(1,a+1):
        for j in range(1,b+1):
            ax=fig.add_subplot(a,b,(i-1)*b+j)
            if c==3:
                ax.imshow(img_arr[(i-1)*b+j-1])
            if c==1:
                ax.imshow(img_arr[(i-1)*b+j-1],cmap='gray')
            ax.axis("off")
            ax.set_title("image - "+str((i-1)*b+j)+ " " +str(np.max(img_arr[(i-1)*b+j-1])) + " " + str(np.min(img_arr[(i-1)*b+j-1])), fontsize=12)
    
    plt.suptitle(text, fontsize=15)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,rect=[0, 0, 1, 0.95])
    plt.savefig(name)
    fig.clf()
    return 

class Pose2Seg(nn.Module):
    def __init__(self):
        super(Pose2Seg, self).__init__()
        self.MAXINST = 8
        ## size origin ->(m1)-> input ->(m2)-> feature ->(m3)-> align ->(m4)-> output
        self.size_input = 512
        self.size_feat = 128
        self.size_align = 64
        self.size_output = 64
        self.cat_skeleton = True

        self.backbone = resnet50FPN(pretrained=True)
        if self.cat_skeleton:
            self.segnet = resnet10units(256 + 55)
        else:
            self.segnet = resnet10units(256)
        self.poseAlignOp = PoseAlign(template_file=osp.dirname(osp.abspath(__file__))+'/templates.json',
                                     visualize=False, factor = 1.0)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.mean = np.ones((self.size_input, self.size_input, 3)) * mean
        self.mean = torch.from_numpy(self.mean.transpose(2, 0, 1)).cuda(0).float()

        self.std = np.ones((self.size_input, self.size_input, 3)) * std
        self.std = torch.from_numpy(self.std.transpose(2, 0, 1)).cuda(0).float()
        self.visCount = 0

        pass

    def forward(self, batchimgs, batchkpts, batchmasks=None):
        self._setInputs(batchimgs, batchkpts, batchmasks)
        self._calcNetInputs()
        self._calcAlignMatrixs()
        output = self._forward()
        # self.visualize(output)
        return output

    def init(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k.replace('pose2seg.seg_branch', 'segnet'): v for k, v in pretrained_dict.items() \
                           if 'num_batches_tracked' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def _setInputs(self, batchimgs, batchkpts, batchmasks=None):
        ## batchimgs: a list of array (H, W, 3)
        ## batchkpts: a list of array (m, 17, 3)
        ## batchmasks: a list of array (m, H, W)
        self.batchimgs = batchimgs
        self.batchkpts = batchkpts
        self.batchmasks = batchmasks
        self.bz = len(self.batchimgs)

        ## sample
        if self.training:
            ids = [(i, j) for i, kpts in enumerate(batchkpts) for j in range(len(kpts))]
            if len(ids) > self.MAXINST:
                select_ids = random.sample(ids, self.MAXINST)
                indexs = [[] for _ in range(self.bz)]
                for id in select_ids:
                    indexs[id[0]].append(id[1])

                for i, (index, kpts) in enumerate(zip(indexs, self.batchkpts)):
                    self.batchkpts[i] = self.batchkpts[i][index]
                    self.batchmasks[i] = self.batchmasks[i][index]


    def _calcNetInputs(self):
        self.inputMatrixs = [translib.get_aug_matrix(img.shape[1], img.shape[0], 512, 512,
                                                      angle_range=(-0., 0.),
                                                      scale_range=(1., 1.),
                                                      trans_range=(-0., 0.))[0] \
                             for img in self.batchimgs]

        inputs = [cv2.warpAffine(img, matrix[0:2], (512, 512)) \
                  for img, matrix in zip(self.batchimgs, self.inputMatrixs)]

        if len(inputs) == 1:
            inputs = inputs[0][np.newaxis, ...]
        else:
            inputs = np.array(inputs)

        inputs = inputs[..., ::-1]
        inputs = inputs.transpose(0, 3, 1, 2)
        inputs = inputs.astype('float32')

        self.inputs = inputs

    def _pose_affinematrix(src_kpt, dst_kpt, dst_area, hard=False):
        src_vis = src_kpt[:, 2] > 0
        dst_vis = dst_kpt[:, 2] > 0
        visI = np.logical_and(src_vis, dst_vis)
        visU = np.logical_or(src_vis, dst_vis)
        # - 0 Intersection Points means we know nothing to calc matrix.
        # - 1 Intersection Points means there are infinite matrix.
        # - 2 Intersection Points means there are 2 possible matrix.
        #   But in most case, it will lead to a really bad solution
        if sum(visI) == 0 or sum(visI) == 1 or sum(visI) == 2:
            matrix = np.array([[1, 0, 0], 
                            [0, 1, 0]], dtype=np.float32)
            score = 0.
            return matrix, score
        
        if hard and (False in dst_vis[src_vis]):
            matrix = np.array([[1, 0, 0], 
                            [0, 1, 0]], dtype=np.float32)
            score = 0.
            return matrix, score
        
        src_valid = src_kpt[visI, 0:2]
        dst_valid = dst_kpt[visI, 0:2]
        matrix = solve_affinematrix(src_valid, dst_valid, fullAffine=False)
        matrix = np.vstack((matrix, np.array([0,0,1], dtype=np.float32)))
        
        # calc score
        #sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        #vars_valid = ((sigmas * 2)**2)[visI]
        vars_valid = 1
        diff = translib.warpAffinePoints(src_valid, matrix) - dst_valid
        error = np.sum(diff**2, axis=1) / vars_valid / dst_area / 2
        score = np.mean(np.exp(-error)) * np.sum(visI) / np.sum(visU)
        
        return matrix, score

    def _calcAlignMatrixs(self):
        ## 1. transform kpts to feature coordinates.
        ## 2. featAlignMatrixs (size feature -> size align) used by affine-align
        ## 3. maskAlignMatrixs (size origin -> size output) used by Reverse affine-align
        ## matrix: size origin ->(m1)-> input ->(m2)-> feature ->(m3(mAug))-> align ->(m4)-> output
        size_input = self.size_input
        size_feat = self.size_feat
        size_align = self.size_align
        size_output = self.size_output
        m2 = translib.stride_matrix(size_feat / size_input)
        m4 = translib.stride_matrix(size_output / size_align)

        self.featAlignMatrixs = [[] for _ in range(self.bz)]
        self.maskAlignMatrixs = [[] for _ in range(self.bz)]
        if self.cat_skeleton:
            self.skeletonFeats = [[] for _ in range(self.bz)]
        for i, (matrix, kpts) in enumerate(zip(self.inputMatrixs, self.batchkpts)):
            m1 = matrix
            # transform gt_kpts to feature coordinates.
            kpts = translib.warpAffineKpts(kpts, m2.dot(m1))

            self.featAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            self.maskAlignMatrixs[i] = np.zeros((len(kpts), 3, 3), dtype=np.float32)
            if self.cat_skeleton:
                self.skeletonFeats[i] = np.zeros((len(kpts), 55, size_align, size_align), dtype=np.float32)

            for j, kpt in enumerate(kpts):
                timers['2'].tic()
                ## best_align: {'category', 'template', 'matrix', 'score', 'history'}
                best_align = self.poseAlignOp.align(kpt, size_feat, size_feat,
                                                    size_align, size_align,
                                                    visualize=True, return_history=False)

                ## aug
                if self.training:
                    mAug, _ = translib.get_aug_matrix(size_align, size_align,
                                                      size_align, size_align,
                                                      angle_range=(-30, 30),
                                                      scale_range=(0.8, 1.2),
                                                      trans_range=(-0.1, 0.1))
                    m3 = mAug.dot(best_align['matrix'])
                else:
                    m3 = best_align['matrix']

                self.featAlignMatrixs[i][j] = m3
                print("image : ", np.shape(self.batchimgs))
                self.maskAlignMatrixs[i][j] = m4.dot(m3).dot(m2).dot(m1)
                print("Align Matrix: ",m3)

                if self.cat_skeleton:
                    # size_align (sigma=3, threshold=1) for size_align=64
                    self.skeletonFeats[i][j] = genSkeletons(translib.warpAffineKpts([kpt], m3),
                                                              size_align, size_align,
                                                              stride=1, sigma=3, threshold=1,
                                                              visdiff = True).transpose(2, 0, 1)
                    print("Skeleton Feats: ", np.shape(self.skeletonFeats))
                    tempname = str(i)+str(j)
                    print(tempname)
                    plot_figure(self.skeletonFeats[i][j], 11, 5, text=tempname)


    def _forward(self):
        #########################################################################################################
        ## If we use `pytorch` pretrained model, the input should be RGB, and normalized by the following code:
        ##      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ##                                       std=[0.229, 0.224, 0.225])
        ## Note: input[channel] = (input[channel] - mean[channel]) / std[channel], input is (0,1), not (0,255)
        #########################################################################################################
        inputs = (torch.from_numpy(self.inputs).cuda(0) / 255.0 - self.mean) / self.std
        [p1, p2, p3, p4] = self.backbone(inputs)
        feature = p1
        feat = feature.clone().detach()
        feat = feat.cpu().numpy()
        # print(feat.device)
        plot_figure2(feat[0],text='')
        print("feature: ",np.shape(feature))
        alignHs = np.vstack(self.featAlignMatrixs)
        indexs = np.hstack([idx * np.ones(len(m),) for idx, m in enumerate(self.featAlignMatrixs)])

        rois = affine_align_gpu(feature, indexs,
                                 (self.size_align, self.size_align),
                                 alignHs)
        print("rois: ",np.shape(rois))
        if self.cat_skeleton:
            skeletons = np.vstack(self.skeletonFeats)
            skeletons = torch.from_numpy(skeletons).float().cuda(0)
            rois = torch.cat((rois, skeletons), 1)

        netOutput = self.segnet(rois)
        # fp = "./r_img/1.jpg"
        # cv2.imwrite(fp, rois)

        if self.training:
            loss = self._calcLoss(netOutput)
            return loss
        else:
            netOutput = F.softmax(netOutput, 1)
            netOutput = netOutput.detach().data.cpu().numpy()
            output = self._getMaskOutput(netOutput)
            # print("visCount: ",self.visCount)
            # if self.visCount%100 == 0:
            self._visualizeOutput(netOutput)
            self.visCount += 1

            return output

    def _calcLoss(self, netOutput):
        mask_loss_func = nn.CrossEntropyLoss(ignore_index=255)

        gts = []
        for masks, Matrixs in zip(self.batchmasks, self.maskAlignMatrixs):
            for mask, matrix in zip(masks, Matrixs):
                gts.append(cv2.warpAffine(mask, matrix[0:2], (self.size_output, self.size_output)))
        gts = torch.from_numpy(np.array(gts)).long().cuda(0)

        loss = mask_loss_func(netOutput, gts)
        return loss


    def _visualizeOutput(self, netOutput):
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

    def _getMaskOutput(self, netOutput):
        netOutput = netOutput.transpose(0, 2, 3, 1)
        MaskOutput = [[] for _ in range(self.bz)]

        idx = 0
        for i, (img, kpts) in enumerate(zip(self.batchimgs, self.batchkpts)):
            height, width = img.shape[0:2]
            for j in range(len(kpts)):
                predmap = netOutput[idx]
                H_e2e = self.maskAlignMatrixs[i][j]

                pred_e2e = cv2.warpAffine(predmap, H_e2e[0:2], (width, height),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)

                pred_e2e = pred_e2e[:, :, 1]
                pred_e2e[pred_e2e>0.5] = 1
                pred_e2e[pred_e2e<=0.5] = 0
                mask = pred_e2e.astype(np.uint8)
                MaskOutput[i].append(mask)

                idx += 1
        return MaskOutput


