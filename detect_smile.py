# import
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils 
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector and the model
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if video path is not given, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping through the frames of the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if reached end of video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert to grayscale and clone
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	# detect faces
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the face bounding boxes
	for (fX, fY, fW, fH) in rects:
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# detect 
		(notSmiling, Smiling) = model.predict(roi)[0]
		label = "Smiling" if Smiling > notSmiling else "Not Smiling"

		# display the label and bounding box on the output image
		cv2.putText(frameClone, label, (fX, fY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
			(0, 0, 255), 2)

	# show output image
	cv2.imshow("Face", frameClone)

	# stopping condition
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# clean up camera and close any open windows
camera.release()
cv2.destroyAllWindows()