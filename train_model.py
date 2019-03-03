# USAGE: python train_model.py

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embedding_path = 'output/embeddings.pickle'
output_recognizer_path = 'output/recognizer.pickle'
output_le_path = 'output/le.pickle'

print("[INFO] loading face embeddings...")
data = pickle.loads(open(embedding_path, "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(output_recognizer_path, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(output_le_path, "wb")
f.write(pickle.dumps(le))
f.close()
