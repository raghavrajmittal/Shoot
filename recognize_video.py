# USAGE: python recognize_video.py
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
from webcam import Webcam
# import arduino_control
from gtts import gTTS
import numpy as np
import subprocess
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

print("[INFO] Smile yo, camera is starting...")
webcam = Webcam()
webcam.start()
time.sleep(2.0)

name_called_out = {}
counts = {}
fps = FPS().start()
old_image = None

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
			print(name)
			# Launch ball when person is close to center of image
			face_center = (startX + endX)/2

			old = counts.get(name, 0)
			new = old + 1
			counts[name] = new

			if counts[name] >= 10 and name_called_out.get(name, False) is False:
				if abs(face_center - w/2) < w/4:
					at_center = True
				if name == "unknown":
					# arduino_control.rotate(arduino)
					tts = gTTS(text="STOP, YOU SHALL NOT PASS.", lang='en')
					tts.save("audio.mp3")
					os.system('mpg321 audio.mp3 -quiet')
					subprocess.call("""ssh pi@143.215.98.197 -t 'python ./Desktop/moveServo.py'""", shell=True)
					filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
					cv2.imwrite('intruders/' + filename, old_image)
				elif name != "unknown":
					tts = gTTS(text="Hey" + name + "!", lang='en')
					tts.save("audio.mp3")
					filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
					cv2.imwrite('cool_peeps/' + filename, old_image)
					os.system('mpg321 audio.mp3 -quiet')
					name_called_out[name] = True
				counts = {}


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
	old_image = frame
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
