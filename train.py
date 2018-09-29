# python train.py --dataset Teste --model dindin.model --labelbin dindin.pickle

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from PIL import Image
from osgeo import gdal

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


EPOCHS = 500
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (7, 55, 55)
# N * M * 7(bandas)
#biblioteca que leia arquivos GEOTIFF
#matriz de confusao, f1 score, precisao, acuracia

data = []
labels = []

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

count = 0
for imagePath in imagePaths:

	raster = gdal.Open(imagePath)
	# print("raster: "+ str(type(raster)))
	arr = raster.ReadAsArray()
	# print("arr: "+ str(arr))
	[bands,cols,rows] = arr.shape
	print("cols: "+ str(cols) + "- rows: "+ str(rows) + "- bands: "+ str(bands))
	trans = raster.GetGeoTransform()
	# print("trans: "+ str(trans))
	proj = raster.GetProjection()
	# print("proj: "+ str(proj))
	# outfile = "outputfile.tif"

	# outdriver = gdal.GetDriverByName("GTiff")
	# print("outdriver: "+ str(outdriver))
	# outdata   = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)
	# print("outdata: "+ str(outdata))
	# teste = outdata.GetRasterBand(1).WriteArray(arr)
	
	# print("teste: "+ str(teste))
    
	count += 1
	print(count)
	data.append(arr)

	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)


aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="reflect")

print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=0)

print("[INFO] serializing network...")
model.save(args["model"])

print(model)
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

