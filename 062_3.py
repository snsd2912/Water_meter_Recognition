from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
from PIL import Image

def is_contour_good(c):
	#approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	return len(approx) == 4
	
# load the example image
image = cv2.imread("6.jpg")
cv2.imshow("image",image)
# get dimensions,height,width of image
dimensions = image.shape
height = image.shape[0]
width = image.shape[1]
condition = 0.5 * width

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map

ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow("eded",thresh2)
th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("edged",th3)
