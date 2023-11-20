import HandRecognition
import pandas as pd
import numpy as np
import cv2
import time 


#seleccionamos la cámara para que nos capture video
cameraCapture = cv2.VideoCapture(0)
#hay que crear una ventana para el capturador
cv2.namedWindow('MyWindow')
cv2.namedWindow('ROI')
print('Showing camera feed.')
dataFrame = pd.DataFrame(data=[], columns=['P1','P2','P3','P4','P5','P6','P7','P8','P9','Gesto'])
print(dataFrame)
img = []
success, frame = cameraCapture.read()
start = time.time()
while success and cv2.waitKey(1) == -1:
    now = time.time()
    while now - start < 5:
        print(now - start )
        cv2.putText(frame,str(now-start),(20,20),fontFace=1,fontScale=20,color=(255,0,0))
        cv2.imshow("MyWindow",frame)
        now = time.time()
        cv2.waitKey(1) == -1
        success, frame = cameraCapture.read()
        #cv2.imshow("ROI",roi)
    frame= cv2.resize(frame,(600,600))
    center = [int(frame.shape[0]/2),int(frame.shape[1]/2)]
    roi = frame[center[1]-250:center[1]+250,center[0]-250:center[0]+250]
    handLandmark,img = HandRecognition.processImage(frame)
    
    while len(handLandmark) < 9:
        handLandmark.append((0,0))

    print(handLandmark)
    lastRowIndex = len(dataFrame)
    handLandmark.append(1)
    print(len(handLandmark))
    if(len(handLandmark) <= 10):
        dataFrame.loc[lastRowIndex] = handLandmark
    #cv2.drawKeypoints(frame,handLandmark,frame,color=(0,0,255))
    cv2.imshow("MyWindow",img)
    cv2.imshow("ROI",roi)
    success, frame = cameraCapture.read()
print(dataFrame)
cv2.destroyWindow('MyWindow')
cameraCapture.release()