# USAGE: python extract_embeddings.py

from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset_path = 'dataset'
output_path = 'output/embeddings.pickle'
detector_path = 'face_detection_model'
embedding_path = 'openface_nn4.small2.v1.t7'
min_confidence = 0.5 	#weak detection threshold

# load our serialized face detector
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face recognizer model
embedder = cv2.dnn.readNetFromTorch(embedding_path)

imagePaths = list(paths.list_images(dataset_path))
knownEmbeddings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > min_confidence:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())

data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(output_path, "wb")
f.write(pickle.dumps(data))
f.close()
