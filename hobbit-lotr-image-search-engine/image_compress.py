import cv2
import glob

dataset_arg = "images/"
dataset_arg2 = "images2/"

args={}
args["dataset"] = dataset_arg
args["dataset2"] = dataset_arg2

i=1
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):

	k = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	resized_image = cv2.resize(image, (400, 166)) 
	cv2.imwrite(args["dataset2"]+str(i)+".jpg",resized_image)
	i+=1