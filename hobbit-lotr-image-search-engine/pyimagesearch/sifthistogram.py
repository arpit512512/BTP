
import numpy as np
import cv2
from collections import Counter

class SIFTHistogram:
	def __init__(self, dictionarySize,dictionary):

		self.dictionarySize = dictionarySize
		self.dictionary = dictionary

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d

	def get_cluster(self,dsc):

		clst = []
		for d in dsc:
			ind=0
			sc=10000000
			cnt=0
			for i in self.dictionary:
			    err = self.chi2_distance(i,d)
			    if(err<sc):
			        sc=err
			        ind = cnt
			    cnt+=1
			clst.append(ind)

		return clst

	def describe(self, image):
		sift = cv2.xfeatures2d.SIFT_create()
		gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, dsc= sift.detectAndCompute(gray, None)
		feature = self.get_cluster(dsc)
		freq = Counter(feature)
		features=[]

		for k in range(self.dictionarySize):
			features.append(freq[k])

		return features