import cv2
import numpy as np
import glob
import os

for file in glob.glob('/Users/mayank/work/techmatters.ai/devpost_challenge/data/raw_input_video/mayank_left_04_29/frames/*png'):
#    file = '/Users/mayank/work/techmatters.ai/devpost_challenge/data/raw_input_video/mayank_left_04_29/frames/out1.png'
    img = cv2.imread(file)
    print(file)
    filename = os.path.basename(file)
    #img = cv3.imread('../../data/raw_input_video/mayank_left_04_29/frames/out44.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(gray, None)

    # pick the keypoint with the largest intensity
    keypoint = None
    max_intensity = 0
    for k in kp:
        if(gray[(int)(k.pt[1]), (int)(k.pt[0])]>max_intensity):
            keypoint=k
            max_intensity= gray[(int)(k.pt[1]), (int)(k.pt[0])]

  #cv2.drawKeypoints(gray,kp,img)
    cv2.circle(img, ((int)(keypoint.pt[0]), (int)(keypoint.pt[1])),  (int)(keypoint.size)+20, (0,255,0))
    cv2.circle(img, ((int)(keypoint.pt[0]), (int)(keypoint.pt[1])),  20, (255,0,0))
    if keypoint == None:
        continue
    
    eye = img[
          (int)(keypoint.pt[1])-(int)(keypoint.size)+20:
          (int)(keypoint.pt[1])+(int)(keypoint.size)+20,
          (int)(keypoint.pt[0])-(int)(keypoint.size)-20:
          (int)(keypoint.pt[0])+(int)(keypoint.size)+20]
    eye = cv2.resize(eye, (96,96))

    cv2.imwrite('./data/frames/' + filename, eye)
    print('./data/frames/' + filename)
#    exit(1)


#cv2.imshow('detected eye',eye)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

