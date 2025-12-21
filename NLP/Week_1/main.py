from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

file = open("data.txt", "r")

# read the data as a string
data = file.read()

# close the file
file.close()

# 1. Normalization : convert the entire in lower case
data_lower = data.lower()

# 2. Noise Removal : removing punctuation
data_cleaned = re.sub(r'[^\w\s]', '', data_lower)

# 3. Tokenization: Split the text into a list of individual words.( also remove whitespaces. )
tokens = word_tokenize(data_cleaned)

# 4. Lemmatization : Reduce tokens to their root forms
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# 5. Stop Word Removal : remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# python dict for counting frequency of each token
token_freq = {}
for token in filtered_tokens:
    if token in token_freq:
        token_freq[token] += 1
    else:
        token_freq[token] = 1
print(token_freq)

input("Press Enter to exit...")

