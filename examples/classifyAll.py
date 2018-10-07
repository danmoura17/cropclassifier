import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
from osgeo import gdal


def Classify(imagaPath):
	raster = gdal.Open(imagaPath)
	image = raster.ReadAsArray()

	image = image.astype("float") / 255.0
	image = np.expand_dims(image, axis=0)
	model = load_model("cropclassifier_cnn_1.00.model")
	lb = pickle.loads(open("CropsLabels.pickle", "rb").read())
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]
	label = "{}: {:.2f}%".format(label, proba[idx] * 100)
	print(imagaPath)
	print(label)
	print("=========")

countjpg = 0
for root, dirs, files in os.walk("."):  
	for filename in files:
		if(filename == "classifyAll.py") or (filename == "cropclassifier_cnn_1.00.model") or (filename == "CropsLabels.pickle"):
			print("Especial")
		else:
			countjpg += 1
			Classify(filename)

