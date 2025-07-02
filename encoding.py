# USAGE
# python encoding.py --dataset images

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True)
args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args["dataset"]))


knownEncodings = []
knownids = []


for (i, imagePath) in enumerate(imagePaths):

	print("Processing image {}/{}".format(i + 1, len(imagePaths)))
	Id = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb, model="hog")

	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:

		knownEncodings.append(encoding)
		knownids.append(Id)


# store the facial encodings and ids to the pickle file
data = {"encodings": knownEncodings, "id": knownids}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

print("Encoding completed successfully")
