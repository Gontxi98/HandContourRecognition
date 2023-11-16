import cv2
import time
import numpy as np
from math import hypot
import HandRecognition
#seleccionamos la cámara para que nos capture video
cameraCapture = cv2.VideoCapture(0)
#hay que crear una ventana para el capturador
cv2.namedWindow('MyWindow')
print('Showing camera feed.')
img = []
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1:
    handLandmark,img = HandRecognition.processImage(frame)
    #cv2.drawKeypoints(frame,handLandmark,frame,color=(0,0,255))
    cv2.imshow("MyWindow",img)
    success, frame = cameraCapture.read()
cv2.destroyWindow('MyWindow')
cameraCapture.release()