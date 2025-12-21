1. read the file(in read mode) in data, then close the file
2. Normalization :- convert entire data into lower case
3. clean data(remove punctuation) using re
4. convert data into tokens using word_tokenize from nltk
5. Lemmentization :- convert words to their root form( not stemming)
6. remove stop word by looping over each token and checking if its stopword(we get set of stopwords using nltk.corpus)
7. make a empty python dictionary and increment count of string(as key) if it is present otherwise set it to one

