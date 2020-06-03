import fitz as pdf
from tfidf_base import TextClassifier 
import os
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
labels = ['Policy' for _ in range(numPolicy)] + ['Claims' for _ in range(numClaims)] + ['Financial' for _ in range(numFinancial)] + ['Medical' for _ in range(numMedical)]
s = [loadText(pdf.open(file)) for file in filenames]
a = TextClassifier(s, labels, 1000)

from flask import Flask, request, make_response, jsonify

app = Flask(__name__)
allowed_extensions = ['pdf']
def allowed_file(filename):
	return '. in filename' and filename.split('.')[-1] in allowed_extensions

@app.route('/classify', methods = ['GET', 'POST'])
def classify():
	if request.method == 'POST':
		if 'pdf-file' not in request.files:
			return 'No pdf found'
		file = request.files['pdf-file']
		if file.filename == '':
			return 'No file selected'
		if file and allowed_file(file.filename):
			file.save('./tempfile.pdf')
			result = a.classify(loadText(pdf.open('./tempfile.pdf')))
			os.remove('./tempfile.pdf')
			return max(result).label



if __name__=='__main__':
	app.run()
