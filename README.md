# PyTF-IDF
Pure Python TF-IDF text classifier

## Prerequisites

No prerequisites, though the testbenches use PyMuPDF for text extraction from sample docs.

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

A model returns scores for each class, compatible with the Python max() and sorted() functions.

```
predictions = model.classify()
result = max(predictions)
print([result.label, result.score])
```
