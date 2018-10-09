import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
from osgeo import gdal


def Classify(imagaPath, model, lb):
	raster = gdal.Open(imagaPath)
	image = raster.ReadAsArray()

	image = image.astype("float") / 255.0
	image = np.expand_dims(image, axis=0)
	
	proba = model.predict(image)[0]
	#print(proba)
	print("\n")
	print("File: "+ str(imagaPath))

	label1 = lb.classes_[0]
	label1 = "{}: {:.2f}%".format(label1, proba[0] * 100)
	print(label1)

	label2 = lb.classes_[1]
	label2 = "{}: {:.2f}%".format(label2, proba[1] * 100)
	print(label2)

	label3 = lb.classes_[2]
	label3 = "{}: {:.2f}%".format(label3, proba[2] * 100)
	print(label3)

	print("\n\nClassified:")

	idx = np.argmax(proba)
	classified = lb.classes_[idx]
	classified = "{}: {:.2f}%".format(classified, proba[idx] * 100)
	print(classified)
	
	
	print("=========")

countjpg = 0
model = load_model("cropclassifier_cnn_1.00.model")
lb = pickle.loads(open("CropsLabels.pickle", "rb").read())
for root, dirs, files in os.walk("."):  
	for filename in files:
		if(filename == "classifyAll.py") or (filename == "cropclassifier_cnn_1.00.model") or (filename == "CropsLabels.pickle"):
			continue
		else:
			countjpg += 1
			Classify(filename, model, lb)



