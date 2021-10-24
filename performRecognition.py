# Import the modules
import cv2
#from sklearn.externals import joblib
import joblib
from skimage.feature import hog
import numpy as np
import imutils
from imutils import contours

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
#im = cv2.imread("photo_1.jpg")
im = cv2.imread("water_meter_cut_bw.png")
# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#cv2.imshow("1",im_gray)
# Threshold the image
#ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#ret, im_th = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY)
#cv2.imshow("2",im_th)
# Find contours in the image
ctrs = cv2.findContours(im_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    # Draw the rectangles
    #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1) 
    # Make the rectangular region around the digit
    #leng = int(rect[3] * 1.6)
    #pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    #pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    x, y, width, height = cv2.boundingRect(rect)
    cv2.rectangle(im, (x,y), (x+width,y+height), (0, 255, 0), 1)
    roi = im_gray[y:y+height, x:x+width]
    #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    #roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    print(str(int(nbr[0])))
    #cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
