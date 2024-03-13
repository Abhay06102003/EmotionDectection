#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow
import keras
import os
import cv2
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import random
from sklearn.model_selection import train_test_split 
from keras.models import Sequential


# In[3]:


new_model = tensorflow.keras.models.load_model("C:\\Users\\Lenovo\\OneDrive\\Desktop\\emotion detector project\\EmotionDetection Trained Model.h5")


# In[4]:


facCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "C:\\Users\\Lenovo\\OneDrive\\Documents\\haarcascade_frontalface_default.cml")


# In[5]:


path = "C:\\Users\\Lenovo\\OneDrive\\Documents\\harcascadeFronface.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (0,0,255)
img = np.zeros((500,500))


# In[6]:


text = 'some text in box'
(text_width,text_height) = cv2.getTextSize(text,font,fontScale = font_scale,thickness = 1)[0]
text_offset_x = 10
text_offset_y= img.shape[0]-25
box_coords = ((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y + text_height+2))


# In[23]:


cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
#cv2.putText(img,(text_offset_x,text_offset_y),font,fontScale = font_scale,color = (0,0,0),thickness = 1)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('CannotOpenWeb')

while(True):
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            print('No face')
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey:ey+eh,ex:ex+ew]
    final_image = cv2.resize(face_roi,(299,299))
    final_image = np.expand_dims(final_image,axis = 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    pred = new_model.predict(final_image)

    if np.argmax(pred) == 0:
        stat = 'Angry'
    elif np.argmax(pred) == 1:
        stat = 'Disgust'
    elif np.argmax(pred) == 2:
        stat = 'Fear'
    elif np.argmax(pred) == 3:
        stat = 'Happy'
    elif np.argmax(pred) == 4:
        stat = 'Sad'
    elif np.argmax(pred) == 5:
        stat = 'Surprise'
    elif np.argmax(pred) == 6:
        stat = 'Neutral'
    status = stat
    x1,y1,w1,h1 = 0,0,175,75
    cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
    cv2.putText(frame,status, (x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    print(stat)
    cv2.imshow('LIVE',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    


# In[ ]:




