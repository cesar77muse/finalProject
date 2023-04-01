# Both V value of HSV and grayscale image can be used for object recognition, but the choice depends on the specific application.

# HSV color space is particularly useful when there is a large variation in lighting conditions. Since the V channel in HSV is based on the brightness of the image, 
# it is less sensitive to variations in lighting than a grayscale image. So, for object recognition in varying lighting conditions,
#  the V value of the HSV color space may be a better choice.

# On the other hand, if the object of interest has a distinct shape or texture, then a grayscale image may be more effective. 
# Grayscale images contain more information about the edges and texture of the object than the V channel in HSV, which only captures brightness.
# So, for object recognition based on shape or texture, a grayscale image may be more appropriate.

import numpy as np
import cv2
  
def on_change(val):
    binary_image = cv2.threshold(img_upscaled, val, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('binary image',binary_image)
    

image = cv2.imread('eye.jpg')

# gray color
imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )
h, s, v = cv2.split(image2)
cv2.imshow('V channel', v)

#upscaling
img_upscaled = cv2.resize(imageG, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# binary
binary_image = cv2.threshold(img_upscaled, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('binary image',binary_image)
cv2.createTrackbar('slider', 'binary image', 0, 255, on_change)

cv2.imshow('gray image upscaled',img_upscaled)

#cv2.imshow('binary image',binary_image)
cv2.waitKey(0)

cv2.destroyAllWindows()