from scipy.spatial import distance as dist
import numpy as np
import cv2
import imutils
from imutils import contours

def order_points(pts):

    xSorted = pts[np.argsort(pts[:, 0])]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

#load image, convert to grayscale, blur it
image = cv2.imread('example.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#find contour in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contour from left to right, initialize color of bounding box
(cnts, _) = contours.sort_contours(cnts, method= 'left-to-right')
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

#loop over the contour
for (i, c) in enumerate(cnts):

    #if the area of contour is small, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour, then
    # draw the contours
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype= 'int')
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    # show the original coordinates
    print("Object #{}:".format(i + 1))
    print(box)

    rect = order_points(box)

    # loop over the original points and draw them
    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    # draw the object num at the top-left corner
    cv2.putText(image, "Object #{}".format(i + 1),
                (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
