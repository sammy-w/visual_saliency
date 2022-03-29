from cv2 import cv2
import os
import skimage.measure 
import numpy as np   
import pandas as pd
import torch

#list video's for analysis
os.chdir('.\\vids\\')
vids = os.listdir()
#get first two numbers of video name and sort in right ordering 1 to 42
vids = list(map(lambda v: float(v[0:2]) , vids ))
vids.sort()
#add .mp4 to filename for indexing
vids = list(map(lambda v: str(v).split('.')[0] + '.mp4' , vids ))
print(vids) #works


#calculate visual entropy and optical flow variance
# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

#jpeg compression size seems to be related to visual complexity
#Dense optical flow, returns matrix of magnitude and direction of motion in video

#add preprocessing (resize images to the same size and blur them to reduce noise (e.g., lighting))
#https://towardsdatascience.com/image-pre-processing-c1aec0be3edf


#image characteristics such as size and sharpness could affect the measurement, maybe we should correct for that
def preprocessing(img):
    #resize frame for faster processing
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    preprocessed_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #blur frames to reduce noise? Alexia said no
   
    return(preprocessed_img)

#create list of dicts for all videos
output_list = []

for vid in vids:
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    print('Processing video: {}'.format(vid))
    
    entropy_list = []
    avg_motion_list = []
    mag_entropy_list = []
    ang_var_list = []
    ang_entropy_list = []

    while(1):
        ret, frame2 = cap.read()
        #break from loop when there are no more frames to process
        if ret == False:
            #calc entropy for last frame
            entropy = skimage.measure.shannon_entropy(prvs)
            entropy_list.append(entropy)
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        #calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        #entropy
        entropy = skimage.measure.shannon_entropy(prvs)
        #average magnitude of motion
        avg_motion = mag.mean()
        #angle of motion variance
        ang_var = ang.var()
        #angle of motion entropy
        ang_entropy = skimage.measure.shannon_entropy(ang)
        #magnitude of motion entropy
        mag_entropy = skimage.measure.shannon_entropy(mag)

        #optional: show output
        #cv2.imshow('Original frame',prvs)
        #cv2.imshow('Dense Optical Flow',bgr)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break

        #update frame
        prvs = next
        #append lists
        entropy_list.append(entropy)
        avg_motion_list.append(avg_motion)
        ang_var_list.append(ang_var)
        ang_entropy_list.append(ang_entropy)
        mag_entropy_list.append(mag_entropy)
    
    #save lists to dict for current video and append dict to output_list (JSON/list of dicts)
    curr_vid = {'video': vid,
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'entropy': entropy_list,
                'avg_motion':avg_motion_list,
                'ang_var': ang_var_list,
                'ang_entropy': ang_entropy_list,
                'mag_entropy': mag_entropy_list
                }
    output_list.append(curr_vid)

#plot
import matplotlib.pyplot as plt
plt.plot(avg_motion_list)
plt.ylabel('Average Motion Magnitude')
plt.show()

plt.plot(ang_entropy_list)
plt.ylabel('Motion Angle Entropy')
plt.show()

plt.plot(entropy_list)
plt.ylabel('Static visual entropy per frame')
plt.show()

plt.plot(mag_entropy_list)
plt.ylabel('Motion Magnitude Entropy')
plt.show()


#possibly do some smoothing based on fps/preprocessing

# turn output into a pickle
import pickle 
  
os.getcwd()
os.chdir('..')
os.getcwd()
  
try: 
    output = open('visual_complexity_data', 'wb') 
    pickle.dump(output_list, output) 
    output.close() 
  
except: 
    print("Something went wrong")


plt.plot(output_list[15]['avg_motion'])
plt.ylabel('Average Motion Magnitude')
plt.show()


plt.plot(output_list[15]['ang_entropy'])
plt.ylabel('Average Motion Angle Entropy')
plt.show()





