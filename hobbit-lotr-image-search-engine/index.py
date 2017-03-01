# author: Adrian Rosebrock
# date: 27 January 2014
# website: http://www.pyimagesearch.com

# USAGE
# python index.py --dataset images --index index.cpickle

# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.sifthistogram import SIFTHistogram
import argparse
import cPickle
import glob
import cv2
import numpy as np

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True,
# 	help = "Path to the directory that contains the images to be indexed")
# ap.add_argument("-i", "--index", required = True,
# 	help = "Path to where the computed index will be stored")
# args = vars(ap.parse_args())

# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}
dataset_arg = "images/"
index_arg = "index.cpickle"
args={}
args["index"] = index_arg
args["dataset"] = dataset_arg
# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])

dictionarySize = 200
BOW = cv2.BOWKMeansTrainer(dictionarySize)
sift = cv2.xfeatures2d.SIFT_create()
## building SIFT dictionary
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):

	k = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	kp, dsc= sift.detectAndCompute(gray, None)
	BOW.add(dsc)

dictionary = BOW.cluster()

desc2 = SIFTHistogram(dictionarySize,dictionary)

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract our unique image ID (i.e. the filename)
	k = imagePath[imagePath.rfind("/") + 1:]

	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = list(desc.describe(image))
	features2 = desc2.describe(image)
	features = features + features2
	# features = features2
	index[k] = np.array(features)

# we are now done indexing our image -- now we can write our
# index to disk
f = open(args["index"], "w")
f.write(cPickle.dumps(index))
f.close()

# show how many images we indexed
print "done...indexed %d images" % (len(index))