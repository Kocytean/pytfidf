import fitz as pdf
from tfidf_base import TextClassifier 

numPolicy = 12
numClaims = 6
numFinancial = 4
numMedical = 7


def loadText(doc):
	allText = ''
	for page in doc:
		allText+=page.getText()
	return allText

# load data 
filenames = ['Data/Policy' + str(i) + '.pdf' for i in range(numPolicy)] + ['Data/Claims' + str(i) + '.pdf' for i in range(numClaims)] + ['Data/Financial' + str(i) + '.pdf' for i in range(numFinancial)] + ['Data/Medical' + str(i) + '.pdf' for i in range(numMedical)]
labels = ['Policy' for _ in range(numPolicy)] + ['Claims' for _ in range(numClaims)]
numDocs = len(labels)
labels += ['Financial' for _ in range(numFinancial)] + ['Medical' for _ in range(numMedical)]
s = [loadText(pdf.open(file)) for file in filenames]
ss = [[page.getText() for page in doc] for doc in [pdf.open(file) for file in filenames]]


numPages = 0
numCorrect = 0
sumDocAccuracy = 0
for i in range(numDocs):

	print('Test ' + str(i) + ': ' + filenames[i])
	
	# instantiate a TextClassifier with training text-label pairs
	a = TextClassifier(s[0:i] + s[i+1:numDocs], labels[0:i] + labels[i+1:numDocs])
	
	t = len(ss[i])
	numPages+=t
	docWeight = 1/t
	docAccuracy=0
	for pagetext in ss[i]:

		#classify a page and grab class prediction with highest score
		result = max(a.classify(pagetext))
		if result.label==labels[i]:
			docAccuracy+=docWeight
			numCorrect+=1
	sumDocAccuracy+=docAccuracy 
	print('Page Classification Accuracy = %' + str(100*docAccuracy))

print('\nPage/document Accuracy: %' + str(100*sumDocAccuracy/numDocs))
print('Page Accuracy: %' + str(100*numCorrect/numPages))