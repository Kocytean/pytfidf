# tfidf_base

Pure Python TF-IDF text classifier

## Prerequisites

The testbenches use PyMuPDF for text extraction from sample docs and a basic Flask server has been provided, but the tfidf_base module has no prerequisites. 

## Usage

### Instantiation

A model is instantiated by feeding a list of documents as strings with a corresponding list of (non-unique) class labels. Optionally, you can provide a length for TF-IDF vectors as well as upper and lower bounds on lengths of words which helps with garbage text often encountered on mass extracting text from PDF sources.

```
from tfidf_base import TextClassifier
model = TextClassifier(list_of_strings,
        list_of_labels,
        vectorSize =1000,
        minWordSize = 5,
        maxWordSize = 25)
```
### Classification

model.classify() returns a list of scores corresponding to each class, compatible with the Python max() and sorted() operations based on score.

```
predictions = model.classify(unclassified_text)
result = max(predictions)
print([result.label, result.score])
```
## Service

Implementation for a basic local Flask server and client communicating on port 5000 has been included.

### Setup

'''
PowerShell:
$env:FLASK_APP='server.py'
flask run

Ubuntu:
export VARNAME="my value"
flask run
'''
