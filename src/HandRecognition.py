import cv2
import numpy as np

img = cv2.imread("./HandContourRecognition/ProjectImages/OpenHand.jpeg")

canny = cv2.Canny(img,100,255)

#img = cv2.GaussianBlur(img,(15,15),10,10)
ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),120, 255, cv2.THRESH_BINARY) 
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 15000:
        print(cv2.contourArea(contour))
        #Con ese con return Points te devuelve directamente los puntos del convex Hull, de la otra manera te devolvería
        hull = cv2.convexHull(contour,returnPoints=False)
        #Este bloque de código sirve para la deteccion del hueco entre dedos
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            cnt = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Teorema del coseno
            if angle <= np.pi / 2:  # Si el angulo es menor que 90 es que hay hueco
                cnt += 1
                cv2.circle(img, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
        img = cv2.drawContours(img,[hull],-1,(0,0,255),2)
        #Este bloque de código sirve para la detección de las puntas de los dedos
        # epsilon = 0.02*cv2.arcLength(c,True)
        # approx = cv2.approxPolyDP(c,epsilon,True)
        # print(approx.size)
        # img = cv2.drawContours(img,[hull],-1,(0,0,255),2)
print(img)