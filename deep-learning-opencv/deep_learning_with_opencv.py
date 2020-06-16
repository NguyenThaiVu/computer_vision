import numpy as np
import cv2

image_path = 'carot1.jpg'
prototxt_path = 'bvlc_googlenet.prototxt'
caffe_model_path = 'bvlc_googlenet.caffemodel'
image_net_label = 'synset_words.txt'

#load image
image = cv2.imread(image_path)

# load the class labels from disk
rows = open(image_net_label).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#load image, resize (224, 224), mean subtraction (104, 117, 123)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

#load model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

#forward through network
net.setInput(blob)
preds = net.forward()

idxs = np.argsort(preds[0])[::-1][:5]

for (i, idx) in enumerate(idxs):
	# draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	# display the predicted label + associated probability to the
	# console
	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))
# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)