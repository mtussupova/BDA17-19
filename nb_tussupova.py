import pandas as pd
import numpy as np
from collections import Counter
from scipy.io import arff

with open("weather.nominal.arff", 'r') as f:
	data, meta = arff.loadarff(f)
with open("soybean.arff", 'r') as f:
	data, meta = arff.loadarff(f)


class Ftr:

	def __init__(self, data, name=None, bin_width=None):
		self.name = name
		self.bin_width = bin_width
		if bin_width:
			self.min, self.max = min(data), max(data)
			bins = np.arange((self.min // bin_width) * bin_width, 
				(self.max // bin_width) * bin_width, bin_width)
			freq, bins = np.histogram(data, bins)
			self.freq_dict = dict(zip(bins, freq))
			self.freq_sum = sum(freq)
		else:
			self.freq_dict = dict(Counter(data))
			self.freq_sum = sum(self.freq_dict.values())
		
	def frequency(self, value):
		if self.bin_width:
			value = (value // self.bin_width) * self.bin_width
		if value in self.freq_dict:
			return self.freq_dict[value]
		else:
			return 0

class NaiveBayes():
	
	def __init__(self, name, *features):
			self.features = features
			self.name = name
			
	def probability(self,ftr_value, ftr): 
  
		if ftr.freq_sum == 0:
			return 0
		else:
			return ftr.frequency(ftr_value) / ftr.freq_sum

class Classificier:
	
	def __init__(self, *nbclasses):
		self.nbclasses = nbclasses
		
		
	def prob(self, *d, best_only=True):
		
		nbclasses = self.nbclasses
		probability_list = []
		for NaiveBayes in nbclasses:            
			ftrs = NaiveBayes.features
			prob = 1
			for i in range(len(ftrs)):
				prob *= NaiveBayes.probability(d[i], ftrs[i])
			  
		probability_list.append( (prob,  NaiveBayes.name) )
		prob_values = [f[0] for f in probability_list]
		prob_sum = sum(prob_values)
		if prob_sum==0:
			number_classes = len(self.nbclasses)
			pl = []
			for prob_element in probability_list:
				pl.append( ((1 / number_classes), prob_element[1]))
			probability_list = pl
		else:
			probability_list = [ (p[0] / prob_sum, p[1])  for p in probability_list]
		if best_only:
			return max(probability_list)
		else:
			return probability_list
