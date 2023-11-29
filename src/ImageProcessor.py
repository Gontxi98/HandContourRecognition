import cv2
import time
import numpy as np
from math import hypot
import HandRecognition
#seleccionamos la cámara para que nos capture video
cameraCapture = cv2.VideoCapture(0)
#hay que crear una ventana para el capturador
cv2.namedWindow('MyWindow')
cv2.namedWindow('ROI')
print('Showing camera feed.')
img = []
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1:
    #frame= cv2.resize(frame,(600,600))
    center = [int(frame.shape[1]/2),int(frame.shape[0]/2)]
    print(center)
    print(frame.shape)
    rect=((center[0]-250,center[1]-250),(center[0]+250,center[1]+250))
    roi = frame[center[1]-250:center[1]+250,center[0]-250:center[0]+250]
    print(rect)

    handLandmark,img = HandRecognition.processImage(frame)
    #cv2.drawKeypoints(frame,handLandmark,frame,color=(0,0,255))
    cv2.rectangle(img,(center[1]-250,center[0]-250),(center[1]+250,center[0]+250),(255,0,0),1)
    cv2.imshow("MyWindow",img)
    cv2.imshow("ROI",roi)
    success, frame = cameraCapture.read()
cv2.destroyWindow('MyWindow')
cameraCapture.release()