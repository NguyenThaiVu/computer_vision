from liveness import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import pickle
import cv2
import os

path_input_dataset = 'dataset'
path_model = 'liveness.model'
path_label_encode = 'le.pickle'
path_output_plot = 'plot.png'

# initialize the initial learning rate, batch size, and number of epochs to train
learning_rate = 0.0001
batch_size = 8
epochs = 50

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
imagePaths = list(paths.list_images(path_input_dataset))
data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    # update the data and labels lists
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array,
# then preprocess it by scaling all pixel intensities to the range [0, 1]

data = np.array(data, dtype= 'float')/255.0

# encode the labels (which are currently strings) as integers and then one-hot encode
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
opt = Adam(learning_rate, decay= learning_rate/epochs)
model = LivenessNet.build(32, 32, 3, classes= len(le.classes_))
model.compile(loss= ['binary_crossentropy'], optimizer= opt, metrics= ['accuracy'])

#train the network
print("[INFO] training network for {} epochs...".format(epochs))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
	epochs=epochs)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(path_model))
model.save(path_model)
# save the label encoder to disk
f = open(path_label_encode, "wb")
f.write(pickle.dumps(le))
f.close()