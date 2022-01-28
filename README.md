# Pose2Seg : Pose-based Human Instance Segmentation Framework

## The Detection Free Human Instance Segmentation pipeline is divided into : 
 - ### Affine-Align Operation
   - Pose Representation : 
     - There are 17 joints (as in COCO- our training dataset), each joint has 3 coordinates - (x,y,v) which are for position in image and visibility. So, each pose is represented by R 17x3 vector. 
     - Distance metric used for K-means is euclidean distance. We only consider poses with more than 8 valid points for K-means clustering. 
     - Mean vector of the class after clustering is taken as the representative element of the class (class template) which is used for pose detection in future
   - Pose Templates : 
     - Training the K-means clustering algo for pose templates:
       - Crop and resize the ROI into unit square, estimate the pose vector (R 17x3 )
       - Apply K-means clustering on R 17x3 vectors.
     - Distance metric for K-means :
       - Euclidean distance
     - Threshold for data points : Poses with more than 8 valid points are considered.
   - Estimate Affine Transformation Matrix : 
     - Finding  best fit affine transform for each of the template poses: H(2x3) is an affine transform for a 2D image. Find the matrix H as, H*(2 x 3) = argmin_H(||H.P - P_μ||)
     - Selecting the template pose with the best score for best fit: Score = exp(-||H*.P - P_μ||)
 - ### Skeleton Features : 
   - There are 55 skeleton features per pose that we make use of while segmentation. These features are of two types
     - Part Affinity fields(PAF) :  They are 2-channel vector field map for each skeleton (line that joins two joints). There are 19 skeletons defined in COCO dataset. Hence for each pose there are 38 channels of PAF features.
     - Confidence maps : Part confidence maps emphasize the importance of those regions around the body part key points (parts = joints). There are 17 joints defined in the COCO dataset Hence, there are 17 channels of part confidence maps.
 - ### SegModule :
   - SegModule is introduced to extend the image features after alignment and is based on the resolution of the aligned RoIs. 

Download pretrained weights and resnet50 pytorch file from below link : https://drive.google.com/file/d/1D9MWWRKiWHyFE6Pm7busmX26oXLYW_Jx/view?usp=sharing
