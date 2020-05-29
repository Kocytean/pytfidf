import fitz as pdf
from tfidf_base import TextClassifier 

numPolicy = 12
numClaims = 6
numFinancial = 4
numMedical = 7


def loadText(doc):
	''' concat text from each page '''
	allText = ''
	for page in doc:
		allText+=page.getText()
	return allText

# load data 
filenames = ['Data/Policy' + str(i) + '.pdf' for i in range(numPolicy)] + ['Data/Claims' + str(i) + '.pdf' for i in range(numClaims)] + ['Data/Financial' + str(i) + '.pdf' for i in range(numFinancial)] + ['Data/Medical' + str(i) + '.pdf' for i in range(numMedical)]
labels = ['Insurance' for _ in range(numPolicy + numClaims)] + ['Financial' for _ in range(numFinancial)] + ['Medical' for _ in range(numMedical)]
s = [loadText(pdf.open(file)) for file in filenames]
numDocs = len(labels)
numCorrect = 0

for i in range(numDocs):

	print('Test ' + str(i) + ': ' + filenames[i])
	
	# instantiate a TextClassifier with training text-label pairs
	a = TextClassifier(s[0:i] + s[i+1:numDocs], labels[0:i] + labels[i+1:numDocs])
	
	#classify complete doc and grab class prediction with highest score
	result = max(a.classify(s[i]))
	print(result, end = ', ')
	if result.label!=labels[i]:
		print('Incorrect')
	else:
		print('Correct')
		numCorrect+=1	


print('Accuracy: %' + str(100*numCorrect/numDocs))
