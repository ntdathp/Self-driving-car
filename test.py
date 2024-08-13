import cv2
import numpy as np
from clear_lane import clear_t, clear_half, clear_top
img = cv2.imread("3_11.jpg")

img = cv2.blur(img,(5,5)) 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (95, 80, 65), (115, 105,82))
mask2 = cv2.inRange(hsv, (95, 27, 115), (115, 70,145))
mask3 = cv2.inRange(hsv, (95, 40, 80), (115, 80,120))

mask = cv2.bitwise_or(mask3, cv2.bitwise_or(mask1, mask2))

img_masked = cv2.bitwise_and(img,img, mask=mask)
gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)	
(thresh, bi_img) = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
img1 = cv2.morphologyEx(bi_img, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((7,7), np.uint8)
img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

kernel = np.ones((11,11), np.uint8)
img3 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

i = img3*1

clear_t(img3,-1,100,0.1)

Hori = np.concatenate((img3, i ), axis=1)
cv2.imwrite('haha.jpg',img3)
cv2.imshow('t',Hori)
cv2.waitKey()