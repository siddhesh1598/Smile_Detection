# DATSET: https://github.com/hromi/SMILEsmileD

# import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

import matplotlib
matplotlib.use("Agg")

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# init list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in paths.list_images(args["dataset"]):
	# load image, preprocess and store in data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=28)
	image = img_to_array(image)
	data.append(image)

	# extract label from the path and store in the labels list
	label = imagePath.split(os.path.sep)[-3]
	label = "smiling" if label == "positive" else "not_smiling"
	labels.append(label)

# normalize the data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the label data
classTotals = labels.sum(axis=0)
print(classTotals)
classWeight = classTotals.max() / classTotals

cW = {}
for i in range(len(classTotals)):
	cW[i] = classWeight[i]

# split the dataset into train/test
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, stratify=labels)

# init the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, 
	classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train 
print("[INFO] training model...")
H = model.fit(trainX, trainY, epochs=15, batch_size=64,
	class_weight = cW, verbose=1)

# evaluate
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=le.classes_))

# save model
print("[INFO] saving model...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()