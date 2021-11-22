
# Importing Necessary libraries

import spacy
from nltk.stem import WordNetLemmatizer         # For proper lemmatization
from nltk.corpus import stopwords   # For removing stopwords
import pandas as pd
import numpy as np
import re

# Importing NLTK libraries

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')


# Importing spacy to achieve name entity rocognition


# Imporing dataset

raw_data = pd.read_csv('C:/Users/dell/Downloads/EnglishTweets.csv')

# Removing the unnecessary column

raw_data.drop('Unnamed: 1', inplace=True, axis=1)

# Printing new table with just text column

print(raw_data.head())

# Text cleaning

stop = stopwords.words('english')
wnl = WordNetLemmatizer()

# Defining a cleantext function


def clean_text(message):
    message = re.sub('[^a-zA-Z]', ' ', message)
    # message = message.lower()
    message = message.split()
    words = [wnl.lemmatize(word) for word in message]
    return  " ".join(words) 

# Implementing the clean_text function to clean our text

raw_data['clean_text'] = raw_data['text'].apply(clean_text)

print(raw_data.head(n=10))


# Defining a function that will return a nameEntity for each clean_text
# Initializing spacy pipeline

nlp = spacy.load('en_core_web_sm')

def nerModel(message):
    text= nlp(message)
    for w in text.ents:
        return("Entities",w.text,w.label_)

        


## Calling the function

raw_data['Extracted_entities'] = raw_data['clean_text'].apply(nerModel)
print(raw_data.head(n=10))

## Exporting into csv file

raw_data.to_csv (r'C:/Users/dell/Desktop/NER_entities_extraction.csv', index = False, header=True)

print (raw_data)

