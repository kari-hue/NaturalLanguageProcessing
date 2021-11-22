# **Project Description**

**Name: Name Entity Recognition using spacy**

### <b> Description of the Project </b>

This project is a simple project where a NER model is construted by using spacy.

### Step-wise implementation:

#### Step 1: Imported necessary libraries and imported the dataset
#### Step 2: Defined a function clean_text() in order to clean the given text.

Code:
```
def clean_text(message):
    message = re.sub('[^a-zA-Z]', ' ', message)
    # message = message.lower()
    message = message.split()
    words = [wnl.lemmatize(word) for word in message]
    return  " ".join(words) 

# Implementing the clean_text function to clean our text

raw_data['clean_text'] = raw_data['text'].apply(clean_text)

print(raw_data.head(n=10))	
```

#### Step 3: Simply used the spacy pipeline to implement NER model using nerModel() function.
Code:
nlp = spacy.load('en_core_web_sm')

def nerModel(message):
    text= nlp(message)
    for w in text.ents:
        return("Entities",w.text,w.label_)
Step 4: Then simply exported the data in the csv file.
Code:
## Exporting into csv file

raw_data.to_csv (r'C:/Users/dell/Desktop/NER_entities_extraction.csv', index = False, header=True)

print (raw_data)



Conclusion: The NER model that I have implemented is a very simple model and it is also not working that amazingly. But still it gives some certain idea about how NER actually works. In future the model can be easily modified and we can build more robust NER model using the concept of transformers and other advanced concept.


