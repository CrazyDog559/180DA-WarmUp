# [180 DA, Lab 1] Python program to
# track an object of certain color in live feed

import cv2
import numpy as np
cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

if (cap.isOpened() == False):
    print("Error reading video file")

while(1):
    # Take each frame
    ret, frame = cap.read()
    if ret == True:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        
        
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        contours,hierarchy = cv2.findContours(mask, 1, 2)
        
        if len(contours) == 0:
            continue
        #cnt = contours[0]

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #cv2.imshow('img',img)
        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask)
        #cv2.imshow('res',res)
        
        # Press S on keyboard to stop the process
        # THE FOCUS SHOULD BE ON THE RECORDING AND NOT THE TERMINAL
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
