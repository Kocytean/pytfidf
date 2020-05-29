import re
from numpy import log
from string import punctuation
from dataclasses import dataclass
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't" "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
table = str.maketrans('','',punctuation + '1234567890“”')
stop_words = [word.translate(table) for word in stop_words]



class FrequencyDict():
	'''Base convenience class for discrete frequency mappings.'''
	def __init__(self, inputKeys = None, return_dict = False):
		self.keys = dict()
		if inputKeys!=None:
			try:
				for key in inputKeys:
					self.add(key)
			except(TypeError):
				self.add(inputKeys)
			if return_dict:
				return self.keys

	def add(self, key):
		if isinstance(key, list):
			for k in key:
				if k in self.keys:
					self.keys[k]+=1
				else:
					self.keys[k]=1
		else:
			if key in self.keys:
				self.keys[key]+=1
			else:
				self.keys[key]=1

	def getRanks(self, n = None):
		#insertion sort
		keys = list(self.keys)
		rankedKeys = [keys[0]]
		for k in keys[1:]:
			looper = True
			i= 0
			while(looper and i<len(rankedKeys)):
				if rankedKeys:
					if self.keys[k] > self.keys[rankedKeys[i]]:
						rankedKeys = rankedKeys[0:i] + [k] + rankedKeys[i:]
						looper = False
					else:
						i+=1

				else:
					rankedKeys = [k]
			if(i==len(rankedKeys)):
				rankedKeys.append(k)
		if n:
			return rankedKeys[:n]
		else:
			return rankedKeys

@dataclass
class Prediction():
	score: float
	label: str
	def __lt__(self, other):
		return self.score<other.score
	def __le__(self, other):
		return self.score<=other.score
	def __gt__(self, other):
		return self.score>other.score
	def __ge__(self, other):
		return self.score>=other.score
	def __eq__(self, other):
		return self.score==other.score
	def __ne__(self, other):
		return self.score!=other.score
		
class TextClassifier():
	""" TF-IDF classifier: initialise with list of texts and labels	"""
	def __init__(self, source=None, labels=None, vectorSize = 1200, minWordSize = 5, maxWordSize = 20):
		'''
		source = list of texts
		labels = list of corresponding label (non-unique)
		vectorSize = number of identifiers with highest tf-idf score to save
		minWordSize = size of smallest key
		maxWordSize = size of largest key
		'''
				
		self.maxWordSize = maxWordSize
		self.minWordSize = minWordSize
		self.classes = dict()
		for l in set(labels):
			self.classes[l]=FrequencyDict()
		self.vocab = FrequencyDict()
		
		for i, text in enumerate(source):
			pageWords = list(filter(lambda x:(len(x)>=minWordSize and len(x)<=maxWordSize) and x not in stop_words,[word.translate(table).lower() for word in text.split()]))
			self.classes[labels[i]].add(pageWords)
			self.vocab.add(pageWords)

		for key in list(self.vocab.keys):
			# print([key,self.vocab.keys[key]])
			self.vocab.keys[key] = 1/log(1+self.vocab.keys[key])
		IDs = list(self.classes)
		for ID in IDs:
			classWords = self.classes[ID].keys
			for key in list(classWords):
				self.classes[ID].keys[key] = log(1+classWords[key])*self.vocab.keys[key]

		self.vectors = dict()
		for ID in IDs:
			self.vectors[ID] = [[identifier, self.classes[ID].keys[identifier]] for identifier in self.classes[ID].getRanks(vectorSize)]

		del self.vocab
		self.classes = IDs
		# for v in list(self.vectors):
		# 	print([v,self.vectors[v]])

	def classify(self, text):

		words = list(filter(lambda x:(len(x)>=self.minWordSize and len(x)<=self.maxWordSize) and x not in stop_words,[word.translate(table).lower() for word in text.split()]))
		wordSet = FrequencyDict(words)
		words = list(wordSet.keys)
		for key in words:
			wordSet.keys[key] = log(1+wordSet.keys[key])
		classScores = []
		for ID in self.classes:
			score = 0
			for identifier in self.vectors[ID]:
				if identifier[0] in words:
					score += identifier[1]*wordSet.keys[identifier[0]]
			classScores.append(Prediction(score, ID))
		return classScores
