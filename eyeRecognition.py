import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

tune1=1.05
tune2=1.05
def nothing(val):
    pass


fps = cap.get(cv2.CAP_PROP_FPS)
windowName = 'frame'
bigWindowName = 'bigger frame'
leftEyeWindowName = 'left eye'
rightEyeWindowName = 'right eye'
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
bigFrame = np.zeros((height*2, width*2), np.uint8)

cv2.namedWindow(bigWindowName)
cv2.createTrackbar('slider', bigWindowName, 0, 255, nothing)



def getBinary(img):
    binary_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('binary image',binary_image)
    
    return img

while True:
    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bigFrame[:height, :width] = gray
    #cv2.putText(frame, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    faces = face_cascade.detectMultiScale(gray,1.3,5) #1.3
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w,y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.04, 4) #1.05
        for (ex, ey, ew, eh) in eyes:
            if ey > h/2.5: # pass if the eye is at the bottom
                pass
            elif ew>90 or eh>90:
                pass
            else:
                if ex < w/2: # left eye
                    rightEye = roi_gray[ey:ey+eh, ex:ex+ew]
                    bigFrame[height:height*2, :width] = cv2.resize(leftEye, (width, height))
                    #cv2.imshow(leftEyeWindowName, leftEye)
                else: #right eye
                    leftEye = roi_gray[ey:ey+eh, ex:ex+ew]
                    bigFrame[height:height*2, width:width*2] = cv2.resize(rightEye, (width, height))
                    binary = cv2.threshold(rightEye,cv2.getTrackbarPos('slider',bigWindowName),255, cv2.THRESH_BINARY)[1]
                    bigFrame[:height, width:width*2] = cv2.resize(binary, (width, height))
                    #cv2.imshow(rightEyeWindowName, rightEye)

    cv2.imshow(bigWindowName, bigFrame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()