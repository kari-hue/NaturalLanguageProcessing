# Project Title: NLP on A Go 

## Description
This project is a simple project that aims in building a application that let's the user to perform common NLP tasks on the input text. The application is demonstrated using streamlit. The features of the application is

* Helps to tokenize given text
* Performs extractive summarization
* Let's you recognize entities
* Let's you discover relationship among the words

Simple look of my project:
![image](https://user-images.githubusercontent.com/57294417/141646074-181ca735-b7d2-4a28-8928-e14df17575bf.png)

I would like to mainly focus on the text summarizer. 

### About the Extractive text summarizer

Stepwise implementation.

#### Step 1: In my text summarizer I have used mainly three techniques for it's implementation:

 * Simply using NLTK library
 * TextRankAlgorithm(Implemented using gensim)
 * LexRankAlgorithm(Implemented using Sumy)

Step 2: Putting everything together using streamlit

After implementing all these algorithms for text summarization I came up with an idea to put everything together using streamlit. Streamlit let’s you build simple and fast applications for demonstration purposes. 

I simply integrated these three summarization techniques in my streamlit application that helps to summarize the text given by the user.

#### Step 3: Finally I evaluated the generated summary using
* Rouge metrcis
* BLeu metrics

Screenshot of implemetation:

#### Text summarization

![image](https://user-images.githubusercontent.com/57294417/141646331-e414c26b-daf5-4c80-839d-f21d85d4ced1.png)

![image](https://user-images.githubusercontent.com/57294417/141646300-f31d9d24-e59a-423f-941c-7d4b6183d469.png)

#### Comparing summary using Rouge metrics

![image](https://user-images.githubusercontent.com/57294417/141646353-4b01d19f-3ff3-4d88-bd10-b54ce28874d4.png)

#### Comparing summary using BLeu

![image](https://user-images.githubusercontent.com/57294417/141646362-7a1b1ab0-db18-4a67-8575-6b13f52c2369.png)


Text summarization is a really vast topic. At the end of the day, we want a summarizer having the ability to summarize the text as we humans do(a system that is able to modify the text, paraphrase, add and remove text, and whatnot). An extractive summarizer doesn’t let you do all that. But if you are just a beginner it’s a good way to start learning how text summarization actually works as extractive summarization is very easy and straightforward to implement. Implementing abstractive summarization is extremely cumbersome and is a topic still under study in the NLP field. 










