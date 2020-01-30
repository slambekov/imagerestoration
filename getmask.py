import cv2
from skimage.measure import compare_ssim
import imutils

import numpy as np



expected_img = cv2.imread("sample/Phase1/Expected/20190105-111821-000000060.jpg")


input_img = cv2.imread("sample/Phase1/input/20190105-111821-000000060.jpg")

expected_img_gray = cv2.cvtColor(expected_img,cv2.COLOR_BGR2GRAY)
input_img_gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(expected_img_gray, input_img_gray, full=True)
diff = (diff * 255).astype("uint8")

diff[diff<240] = 0
diff = ~diff
kernel = np.ones((3,3),np.uint8)
diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

diff = cv2.dilate(diff,kernel,iterations = 1)



ratio = 10
shape_expect = int(expected_img.shape[0]/ratio),int(expected_img.shape[1]/ratio)
shape_input = int(input_img.shape[0]/ratio),int(input_img.shape[1]/ratio)
shape_diff = int(diff.shape[0]/ratio),int(diff.shape[1]/ratio)


# print(shape_expect)
# exit(0)
expected_img = cv2.resize(expected_img,(shape_expect[1],shape_expect[0]))
input_img = cv2.resize(input_img,(shape_input[1],shape_input[0]))
diff = cv2.resize(diff,(shape_diff[1],shape_diff[0]))

cv2.imwrite("input.jpg",input_img)
cv2.imwrite("mask.jpg",diff)
exit(0)
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ






# thresh = cv2.threshold(diff, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow("thresh1",diff.copy())

# cnts = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# # print((cnts))

# cv2.drawContours(diff, cnts, 1, (255,255,255), -1)



cv2.imshow("expected_img",expected_img)
cv2.imshow("input_img",input_img)
cv2.imshow("thresh",diff)
cv2.waitKey(0)


