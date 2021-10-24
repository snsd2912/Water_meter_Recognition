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
image = cv2.imread("water_meter.jpg")
cv2.imshow("image",image)
# get dimensions,height,width of image
dimensions = image.shape
height = image.shape[0]
width = image.shape[1]
condition = 0.5 * width

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 60, 200,255)
cv2.imshow("edged",edged)
#cv2.imshow('water_meter_canny.jpg',edged)
cnts = cv2.findContours(edged.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
mask = np.zeros(image.shape, dtype="uint8")

#remove unwanted contours
for c in cnts:
    if cv2.arcLength(c,False)<condition:
        cv2.drawContours(mask, [c], 0, (255,255,255,255), 1)
#cv2.imshow('water_meter_canny_copy.jpg',mask)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
edged = cv2.bitwise_xor(mask, edged)
cv2.imshow("final_edged",edged)
#the second morphology
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cnts = cv2.findContours(edged.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
frame = None
max = 0.0
for c in cnts:
        if is_contour_good(c):
                peri = cv2.arcLength(c, True)
                if peri > max:
                        max = peri
                        frame = c
#cv2.drawContours(image, [frame], 0, (0,255,0), 3)
x, y, width, height = cv2.boundingRect(frame)
#cv2.imshow("we_got_this",image)
roi = image[y:y+height, x:x+width]
cv2.imwrite("water_meter_cut.png", roi)
