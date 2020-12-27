"""
Image Stitching Problem
(Due date: Nov. 9, 11:59 P.M., 2020)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
   
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    sift = cv2.xfeatures2d.SIFT_create()
    
    img1 =right_img
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img1_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = left_img
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

    img2_1 =cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    IMG1=img1.copy()
    IMG2=img2.copy()
    
    gray1=img1.copy()
    gray1[0:800,800:1300]=1
    kp1, des1 =sift.detectAndCompute(gray1,None)

    gray2=img2.copy()
    gray2[0:800,0:650]=1
    kp2, des2 =sift.detectAndCompute(gray2,None)

    gray1[0:800,800:1300]=img1[0:800,800:1300]
    img_right=cv2.drawKeypoints(gray1,kp1,IMG1)

    gray2[0:800,0:650]=img2[0:800,0:650]
    img_left=cv2.drawKeypoints(gray2,kp2,IMG2)
    
    dict1,dict2={},{}
    for i in range(0,len(kp1)):
        dict1[kp1[i]]=des1[i]
    
    for i in range(0,len(kp2)):
        dict2[kp2[i]]=des2[i]
    
    dict3={}
    for i in dict1:
        for j in dict2:
            dict3[i,j]=np.linalg.norm(dict1[i]-dict2[j])
        
    sorted_dict3 = sorted(dict3.items(), key=lambda kv: kv[1])
    
    dict_final=sorted_dict3[0:110]
    matched_features1,matched_features2=[],[]
    for i in dict_final:
        matched_features1.append(i[0][0])
        matched_features2.append(i[0][1])
      
    matched_features11 = np.float32([np.float32(matched_features1[pos].pt) for pos in range(0, len(matched_features1))]).reshape(-1,1,2)
    matched_features21 = np.float32([np.float32(matched_features2[pos].pt) for pos in range(0, len(matched_features2))]).reshape(-1,1,2)
    homography, mask = cv2.findHomography(matched_features11,matched_features21,cv2.RANSAC,ransacReprojThreshold=10.0)
    
    final_image = cv2.warpPerspective(img1,homography,(img1.shape[1]+img1.shape[1],img2.shape[0]))
    final_image[0:img2.shape[0],0:img2.shape[1]] = img2
#plt.figure(figsize=(15,15))
#plt.imshow(final_image)

    def touchup(im):
    
        if not np.sum(im[0]):
            return touchup(im[1:])
    
        if not np.sum(im[-1]):
            return touchup(im[:-2])
    
        if not np.sum(im[:,0]):
            return touchup(im[:,1:])
    
        if not np.sum(im[:,-1]):
            return touchup(im[:,:-2])
        return im

    k=touchup(final_image)
    finale=cv2.cvtColor(k,cv2.COLOR_BGR2RGB)

    return finale



    raise NotImplementedError

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg',result_image)


