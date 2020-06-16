import numpy as np
import cv2
import os

path_input_video = 'videos/real.mov'
path_output_directory_crop_face = 'dataset/real'
path_face_detector_model = 'face_detector'
threshold = 0.5
frame_skip = 4

#load face detector model from disk
proto_path = path_face_detector_model + '/deploy.prototxt'
model_path = path_face_detector_model + '/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(path_input_video)
read = 0
saved = 0

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    read += 1

    if read % frame_skip != 0:
        continue

    # grab the frame dimension and construct a blob from the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # write the frame to disk
            p = os.path.sep.join([path_output_directory_crop_face, "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()


print(saved)