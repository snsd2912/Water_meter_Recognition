from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog

def is_contour_good(c):
	#approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	return len(approx) == 4
	
# load the example image
image = cv2.imread("water_meter.jpg")
#cv2.imshow("image",image)
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
x, y, width, height = cv2.boundingRect(frame)
roi = image[y:y+height, x:x+width]
cv2.imwrite("water_meter_cut.png", roi)
'''
img = cv2.imread("water_meter_cut.png")
dim = (200,40)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_img,110,255,cv2.THRESH_BINARY)
cv2.imwrite("water_meter_cut_bw.png",thresh)
cv2.imshow("2",thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if w >= 5 and (h >= 15 and h <= 20):
		digitCnts.append(c)
digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

i=0
for c in digitCnts:
        x, y, width, height = cv2.boundingRect(c)
        cv2.rectangle(thresh, (x,y), (x+width,y+height), (255, 255, 255), 1)
        i = i+1
        string = "water_meter_cut_"+str(i)+".png"
        x, y, width, height = cv2.boundingRect(c)
        roi = thresh[y:y+height, x:x+width]
        cv2.imwrite(string, roi)
        old_im = Image.open(string)
        old_im = old_im.resize((14,14))
        old_size = old_im.size
        new_size = (28, 28)
        new_im = Image.new("RGB", new_size)
        new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),int((new_size[1]-old_size[1])/2)))#        new_im.save(string)
'''
                
# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
#im = cv2.imread("photo_1.jpg")
im = cv2.imread("water_meter_cut.png")
# Convert to grayscale and apply Gaussian filtering
dim = (200,40)
im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(im_gray,110,255,cv2.THRESH_BINARY)
ctrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
ctrs = imutils.grab_contours(ctrs)
digitCnts = []
for c in ctrs:
	(x, y, w, h) = cv2.boundingRect(c)
	if (w >= 5 and w <= 15) and (h >= 15 and h <= 25):
		digitCnts.append(c)
digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
rects = [cv2.boundingRect(ctr) for ctr in digitCnts]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in digitCnts:
    x, y, width, height = cv2.boundingRect(rect)
    cv2.rectangle(im, (x,y), (x+width,y+height), (0, 255, 0), 1)
    roi = thresh[y:y+height, x:x+width]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    print(str(int(nbr[0])))
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

