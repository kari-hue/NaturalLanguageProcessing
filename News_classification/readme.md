## Problem statement:

This is a multiclass classification problem in which given a text we need to predict in which class of news it belongs to. There are altogether 5 news classes (Business,Tech,Government,Sports and Entertainment)
For this text classification problem, I simply used multinomial Bayes theorem to train my model.

### Libraries used :
* Pandas
* Numpy
* Matplotlib
* pyplot
* NLTK
* Sklearn

### Stepwise Implementation of the classification problem:

#### Step 1: Importing all the necessary libraries

#### Step 2: Importing the data and doing some exploratory analysis
![image](https://user-images.githubusercontent.com/57294417/142883812-93ce76db-2d4e-4c04-8f84-fb574bb243df.png)

#### Step 3: Declare a function that would do text cleaning if required.
```
# Cleaning the text

# Defining a function

def clean_text(message):
    message = re.sub('[^a-zA-Z]',' ',message)
    message = message.lower()
    message = message.split()
    words = [wnl.lemmatize(word) for word in message if word not in stop]
    
    return  " ".join(words)


news_df["clean_text"] = news_df["Text"].apply(clean_text)
news_df.head(n=10)

```
#### Step 4:  Did some text exploration where I extracted keywords from the given text for different Genre of news and the draw a wordCloud out of the information.

Function for Extracting keywords:

```
def extract_keywords(text,num = 50):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)
```
 
Generating wordcloud from the keywords demonstration:
For tech news:
![image](https://user-images.githubusercontent.com/57294417/142884159-518cd14f-ff22-48c4-8886-ff51b1c45eb2.png)

 For government news
 ![image](https://user-images.githubusercontent.com/57294417/142884231-393362ae-1a46-49d3-9271-14b5d74e609f.png)

 So on and so forth……….

#### Step 5: Did vectorization of the text using Tfidf vectorizer. Encoded the output label using LabelEncoder()

Vectorization:
```
# Implementing the vectorizer

vectorization = TfidfVectorizer(binary = False,use_idf = True)
xv_train = vectorization.fit_transform(X)
#xv_test = vectorization.transfrom(X_test)
```
 
LabelEncoding:
 
I encoded the data simply to represent by output label in numbers. And the encoding resulted in representing
0 - Business 
1 - entertainment 
2 - politics 
3 - sport 
4 – tech

#### Step 6: Splitted the dataset into train and test data and then applied multinomial Naïve Bayes algorithm to build a model. After training the model for the training dataset we simply predicted our models performed on the test dataset.

 
#### Step 7: Used Evaluation metrics to measure the performance of the model
Accuracy:
 ![image](https://user-images.githubusercontent.com/57294417/142884979-928da111-1969-450c-81ee-4485e59d74b4.png)

 Confusion matrix:
 
 ![image](https://user-images.githubusercontent.com/57294417/142884471-6c01629e-682f-43b8-bd92-33546a49d69d.png)

Classification Report
![image](https://user-images.githubusercontent.com/57294417/142884567-8aed391e-16d6-4ba8-bda5-68bf086380a0.png)
 
Roc-Curve
```
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 5

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, predict_proba_news1[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='red', label='Class 4 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);   
```
![image](https://user-images.githubusercontent.com/57294417/142884767-c9d84fc0-1c71-4b44-acff-27b3755ed40b.png)

Drawing a ROC-Curve for the binary classification problem is pretty simple but in case of multiclass classification problem you can use the concept of “ovr” or “ovo” to construct a ROC Curve.
 
#### Step 8: Finally defining a function that will take a text and help predict the news class.

Function:
```
## Defining a function that will predict the certain type of the news.

def predict_news(text,model):
    myvect = vectorization.transform(text).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_,pred_proba))
    print(prediction)
    return pred_percentage_for_all

 ```

Calling the function on the following text:
predict_news(['It is believed that millions of cybercrime occurs everyyear in the east coast of virginia due to these crime the government is passing laws and making reforms on better computer security.'],Dt)

Output:
![image](https://user-images.githubusercontent.com/57294417/142884884-dee46787-f5f5-4279-a35a-5b7d1721a46b.png)

 Predicated as government news.


