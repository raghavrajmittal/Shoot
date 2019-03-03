# USAGE: python recognize_video.py

from imutils.video import VideoStream
from imutils.video import FPS
from webcam import Webcam
# import arduino_control
import numpy as np
import imutils
import pickle
import time
import cv2
import os

detector_path = 'face_detection_model'
embedding_path = 'openface_nn4.small2.v1.t7'
recognizer_path = 'output/recognizer.pickle'
le_path = 'output/le.pickle'
min_confidence = 0.5

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_path)
recognizer = pickle.loads(open(recognizer_path, "rb").read())
le = pickle.loads(open(le_path, "rb").read())
#
# arduino = None
# print("[INFO] Connecting arduino...")
# try:
# 	arduino = arduino_control.connect()
# 	print(arduino)
# except Exception:
#     print("closing connection")
#     if arduino:
#         arduino_control.disconnect(arduino)

print("[INFO] Smile yo, camera is starting...")
webcam = Webcam()
webcam.start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

while True:
	frame = webcam.get_current_frame()

	# frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > min_confidence:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# Launch ball when person is close to center of image
			face_center = (startX + endX)/2
			if abs(face_center - w/2) < w/4:
				at_center = True
			if name is 'Unknown' and at_center:
				# arduino_control.rotate(arduino)
				pass

			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
