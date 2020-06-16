import face_recognition
import cv2
import numpy as np

image = face_recognition.load_image_file("0001.png")
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_location

    # You can access the actual face itself like this:
    face_image = np.array(image[top:bottom, left:right])
    cv2.imshow('carot', face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
