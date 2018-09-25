# python train.py --dataset Teste --model dindin.model --labelbin dindin.pickle

import argparse
from imutils import paths
from osgeo import gdal
import cv2
import random


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
IMAGE_DIMS = (96, 96, 3)
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

	image = cv2.imread(imagePath)
	raster = gdal.Open(imagePath)
	print(type(raster))
	#
	print(imagePath)
	count += 1
	print (count)

	