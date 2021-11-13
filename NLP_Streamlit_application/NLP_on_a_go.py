# python packages


import streamlit as st
import numpy as np
import pandas as pd
import re
from rouge import Rouge
from rouge import FilesRouge
from nltk.translate.bleu_score import sentence_bleu
import argparse

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob.blob import Sentence
import nltk
from gensim.summarization import summarize

import spacy
import spacy_streamlit

from spacy import displacy
from textblob import TextBlob
nlp = spacy.load('en_core_web_sm')


# nltk packages

# sumy packages


# NLP Pkg


# HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# gensim pkg

# nltk packages
nltk.download('stopwords')
nltk.download('punkt')          # The Punkt sentence tokenizer.

# sumy pkg


# function for text analyzer
def text_analyzer(my_text):

    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = ['"Tokens" : {},\n"Lemma":{}'.format(
        token.text, token.lemma_) for token in docx]
    return allData

# summarizer using nltk


def nltk_Summarizer(text):

    stopWords = set(stopwords.words('english'))
    words = word_tokenize(text)

    freqtable = dict()

    for word in words:
        word = word.lower()
        if word in stopWords:
            continue

        if word in freqtable:
            freqtable[word] += 1
        else:
            freqtable[word] = 1

    # Creating a dictionary to keep the core

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqtable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += 1
                else:
                    sentenceValue[sentence] = freq

    sumvalues = 0

    for sentence in sentenceValue:
        sumvalues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumvalues/len(sentenceValue))

    # Storing sentence in our summary

    summary = ' '

    for sentence in sentences:

        if(sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    return(summary)


# Sumy Summarizer function

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# Function that will implement rogue metrics

def rogueEvaluation(hypothesis, reference):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(hypothesis, reference)
    return(scores)


## Function that will implement Bleu Metrics
def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str, default='newsummary.txt', help='Reference File')
    Argparser.add_argument('--candidate', type=str, default='ref.txt', help='Candidate file')

    args = Argparser.parse_args()
    return args



# @st.cache(allow_output_mutation=True)


def main():
    """NLP App with streamlit"""
    st.title("Text Summarizer with Streamlit")
    st.subheader("Natural language processing on the GO")

    activities = ["Tokens", "Summarizer", "NER", "Relationship"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # Tokenization
    if choice == "Tokens":
        st.subheader("Tokenize you Text")
        message = st.text_area("Enter your Text", "Type Here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Text summarization
    if choice == "Summarizer":
        st.subheader("Summarize your text")
        message = st.text_area("your text here", "Type here")
        summary_options = st.selectbox(
            "Choose your summarizer", ("nltk", "gensim", "sumy"))

        text_range = st.sidebar.slider(
            "Summarize words Range, For Gensim only", 25, 500)

        if st.button("summarize"):
            if summary_options == 'nltk':

                summary_result = nltk_Summarizer(message)

            elif summary_options == 'gensim':
                st.text("Using gensim....")
                summary_result = summarize(message, word_count=text_range)

            elif summary_options == 'sumy':
                st.text("Using Sumy....")
                summary_result = sumy_summarizer(message)

            else:
                st.warning("Using Default summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message, word_count=text_range)
            
            st.success(summary_result)
           
            generated_summary = summary_result
            file = open("newsummary.txt","w") 
            file.write(generated_summary) 
           

            #reference_text = st.text_area("Type here","Type here")

            #if st.button("Evaluate"):
                #reference_summary = reference_text
                
                #file1 = open("reference.txt","a")
                #file1.write(reference_summary)


        ##Entering the human generated summary
        st.subheader("Human generated summary")
        ref_text = st.text_area("Enter your Text", "Type Here")
        file1 = open("ref.txt","w")  ## Store the text in the textbox into ref.txt file
        file1.write(ref_text) 
        file1.close()
        
        if st.button("Evaluate score with the human summary(Using Rogue Metrics)"):

            hyp_path = "newsummary.txt"
            ref_path = "ref.txt"
           
                
            scores = rogueEvaluation(hyp_path, ref_path)   ## Calling the rogueEvaluation function
            st.write(scores)
        
        if st.button("Evaluate score with the human summary(Using BLeu)"):
            args = argparser()
            reference = open(args.reference, 'r').readlines()
            candidate = open(args.candidate, 'r').readlines()
            if len(reference) != len(candidate):
                raise ValueError('The number of sentences in both files do not match.')
            score = 0.
            for i in range(len(reference)):
                    score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                    score /= len(reference)
                    st.write("The bleu score is: "+str(score))


           

    if choice == "NER":
        st.subheader('Named Entity Recognizer')
        raw_docx = st.text_area('Your Text')
        docx = nlp(raw_docx)
        if st.button('Analyze'):
            spacy_streamlit.visualize_ner(
                docx, labels=nlp.get_pipe("ner").labels)

    if choice == "Relationship":
        st.subheader("Extract the relationship between the text")
        message = st.text_area("Enter your Text", "Type Here")
        docx = nlp(message)
        if st.button('Analyze'):
            spacy_streamlit.visualize_parser(docx)


# Named Entity
if __name__ == '__main__':
    main()
