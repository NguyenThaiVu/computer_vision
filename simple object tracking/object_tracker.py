import centroid_tracker
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

prototxt_path = 'deploy.prototxt'
caffe_model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
threshold = 0.5

ct = centroid_tracker.CentroidTracker()
(H, W) = (None, None)

net = cv2.dnn.readNetFromCaffe(prototxt= prototxt_path, caffeModel= caffe_model_path)

vs = VideoStream(src= 0).start()

#loop over the frame from video stream
while True:

    #read frame from video and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width= 400)

    #if the frame dimension are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Ca rot", frame)
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()