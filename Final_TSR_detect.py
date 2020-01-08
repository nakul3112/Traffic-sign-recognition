# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:16:40 2019

@author: nakul
"""

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from sklearn import datasets
from sklearn import svm


def getSignPrediction(prediction):
    signImg = 0
    if prediction == 1 :
        signImg = 0
    elif prediction == 14:
        signImg = 1
    elif prediction == 17:
        signImg = 2
    elif prediction == 19:
        signImg = 3
    elif prediction == 21:
        signImg = 4
    elif prediction == 35:
        signImg = 5
    elif prediction == 38:
        signImg = 6
    else: 
        signImg = 7
        
    return signImg


imgList = glob.glob("/media/nakul/DATA/Ubuntu_UMD_Courses/Perception/Project-6/TSR/input/*")
imgList.sort()
#imgList = imgList[80:]
hog = cv2.HOGDescriptor()

filename = open("dataset.pkl",'rb')
clf = pickle.load(filename)

tmpl = glob.glob("templates/*.png")
outImg = cv2.imread(imgList[0])
out = cv2.VideoWriter('NIA.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (outImg.shape[1],outImg.shape[0]))

for img in imgList:
    regions=[] 
    # ======== Traffic sign detection using HSV ===================== #
    #image = cv2.imread("image.032722.jpg")
    image = cv2.imread(img)
    #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    #/media/nakul/DATA/Ubuntu_UMD_Courses/Perception/Project-6/templates
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    # For Red
    mask_r1 = cv2.inRange(hsv, np.array([0, 70, 100]), np.array([15, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 70, 100]), np.array([180, 255, 255]))
    dst = cv2.addWeighted(mask_r1,1.0,mask_r2,1.0,0)
    blur = cv2.GaussianBlur(dst,(5,5),0)
    _,contours,hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #print('ar', cv2.contourArea(cnt))
        if w/h >= 1.0 and w/h <= 2.0 and cv2.contourArea(cnt) >= 400 and cv2.contourArea(cnt) <= 3000:
            #red1 = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            
            #####
            #regions.append(image[y: y+h, x: x+w])
            resized = cv2.resize(np.array(image[y: y+h, x: x+w]), (128, 128)) 
            res_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            
            hog_feat = hog.compute(res_gray)
            hog_feat = hog_feat.reshape(1, -1)
           
            y_pred = clf.predict(hog_feat)
            #print("Check", int(y_pred[0]))
            indices = getSignPrediction(int(y_pred[0]))
                      
            template = cv2.imread(tmpl[indices]) 
            #cv2.imshow("template", template)
            #tmpl = cv2.resize(np.array(template),(128,128),interpolation = cv2.INTER_CUBIC)
            class_probabilities = clf.predict_proba(hog_feat)
            #print("ClassProb  ", class_probabilities)
            if np.amax(class_probabilities) >= 0.90:
                red1 = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                if x>64 and y < image.shape[0] - 64 :                
                    image[y: y+64, x-64: x] = template             
                else:
                    try:
                        image[y: y+64, x+64: x + (2*64)] = template
                    except:
                        pass       
     
                
    # For Blue            
    mask_b = cv2.inRange(hsv, np.array([100,45,50]), np.array([130,250,250]))       
    blur_b = cv2.GaussianBlur(mask_b,(5,5),0)
    _,contours,hierarchy = cv2.findContours(blur_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #print('ar', w/h)
        if w/h >= 0.5 and w/h <= 1.5 and cv2.contourArea(cnt) >= 400 and cv2.contourArea(cnt) <= 4000:
            #blue = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            
            #####
            #regions.append(image[y: y+h, x: x+w])
            resized = cv2.resize(np.array(image[y: y+h, x: x+w]), (128, 128)) 
            res_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            hog_feat = hog.compute(res_gray)
            hog_feat = hog_feat.reshape(1, -1)
            
            y_pred = clf.predict(hog_feat)
            indices = getSignPrediction(int(y_pred[0]))
            
            template = cv2.imread(tmpl[indices]) 
            #cv2.imshow("template", template)
            #tmpl = cv2.resize(np.array(template),(128,128),interpolation = cv2.INTER_CUBIC)
            class_probabilities = clf.predict_proba(hog_feat)
            #print("ClassProb  ", class_probabilities)
            if np.amax(class_probabilities) >= 0.90:
                blue = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                if x>64 and y < image.shape[0] - 64 :                
                    image[y: y+64, x-64: x] = template             
                else:
                    try:
                        image[y: y+64, x+64: x + (2*64)] = template
                    except:
                        pass

    #cv2.imshow('Image',blur_b)
    #print('image_shape',image.shape[0])
    out.write(image)
    cv2.imshow('Result_Signs',image)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#cv2.waitKey(0)
cv2.destroyAllWindows()







