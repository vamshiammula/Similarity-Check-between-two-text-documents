# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:10:08 2020

@author: VAMSHI
"""


import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
import re
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.summarizer import summarize
from nltk.sentiment.vader import SentimentIntensityAnalyzer



data = open("about_masks.txt","r",encoding="utf8").read()

data2 = open("guidelines_masks.txt","r", encoding="utf8").read()

data3 = open("C:/Users/VAMSHI/Documents/standard_documents/covid.txt","r").read()

text = summarize(data, ratio=0.05)
text


def process_sum(row):
     data = row

     #Removes unicode strings like "\u002c" and "x96"
     data = re.sub(r'(\\u[0-9A-Fa-f]+)',r" ", data)
     data = re.sub(r'[^\x00-\x7f]',r" ",data)
     #Remove additional white spaces
     data = re.sub('[\s]+', ' ', data)
     data = re.sub('[\n]+', ' ', data)
     row = data
     return row
#call the function with your data
data_sum = process_sum(text)

data_sum

def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    print('Positive polarity score:',str(round(float(polarity_scores['pos'])*100,2))+'%')
    print('Negative polarity score:',str(round(float(polarity_scores['neg'])*100,2))+'%')
    if polarity_scores['neg'] > polarity_scores['pos']:
        return 'negative'
    else:
        return 'positive'

fetch_sentiment_using_SIA(data)

fetch_sentiment_using_SIA(data2)
fetch_sentiment_using_SIA(data3)


# Standardizing Data

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

d = {'WHO':'World Health Organization','COVID-19':'Covid19'}
data = replace_all(data, d)

def processRow(row):
     data = row
     #Lower case
     data = data.lower()
     #Removes unicode strings like "\u002c" and "x96"
     data = re.sub(r'(\\u[0-9A-Fa-f]+)',r" ", data)
     data = re.sub(r'[^\x00-\x7f]',r" ",data)
     #convert any url to URL
     data = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',data)
     #Convert any @Username to "AT_USER"
     data = re.sub('@[^\s]+','AT_USER',data)
     #Remove additional white spaces
     data = re.sub('[\s]+', ' ', data)
     data = re.sub('[\n]+', ' ', data)
     #Remove not alphanumeric symbols white spaces
     data = re.sub(r'[^\w]', ' ', data)
     #Removes hastag in front of a word """
     data = re.sub(r'#([^\s]+)', r'\1', data)
     #Replace #word with word
     data = re.sub(r'#([^\s]+)', r'\1', data)
     #Remove :( or :)
     data = data.replace(':)',"")
     data = data.replace(':(',"")
     #remove multiple exclamation
     data = re.sub(r"(\!)\1+", ' ', data)
     #remove multiple question marks
     data = re.sub(r"(\?)\1+", ' ', data)
     #remove multistop
     data = re.sub(r"(\.)\1+", ' ', data)
     #lemma
     data =" ".join([Word(word).lemmatize() for word in data.split()])
     #stemmer
     #st = PorterStemmer()
     #data=" ".join([st.stem(word) for word in data.split()])
     #Removes emoticons from text
     data = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', "", data)
     #trim
     data = data.strip('\'"')
     # tokens of words  
     data = sent_tokenize(data)
     
     row = data
     return row
#call the function with your data
data = processRow(data)

#spelling correction 
correct_words = []

spell = Speller(lang='en')
for i in data:
    k = spell(i)
    correct_words.append(k)

data = pd.DataFrame({'text':correct_words})
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['text']

corpus = data['text'].to_list()


data2 = replace_all(data2, d)

data2 = processRow(data2)

#spelling correction 
correct_words2 = []

spell = Speller(lang='en')
for i in data2:
    k = spell(i)
    correct_words2.append(k)


data2 = pd.DataFrame({'text':correct_words2})
stop = stopwords.words('english')
data2['text'] = data2['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data2['text']


corpus2 = data2['text'].to_list()

train_set = corpus

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
tfidf_matrix_test1 = tfidf_vectorizer.transform(corpus2)

data3 = replace_all(data3, d)

data3 = processRow(data3)

#spelling correction 
correct_words3 = []

spell = Speller(lang='en')
for i in data3:
    k = spell(i)
    correct_words3.append(k)


data3 = pd.DataFrame({'text':correct_words3})
stop = stopwords.words('english')
data3['text'] = data3['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data3['text']


corpus3 = data3['text'].to_list()

list_corpus = [corpus,corpus2,corpus3]

train_set = corpus

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
tfidf_matrix_test2 = tfidf_vectorizer.transform(corpus3)

siml_data2 = str(round(float(cosine_similarity(tfidf_matrix_train,tfidf_matrix_test1))*100,2))+'%'
print('similarity between standard document and trail document1 is',siml_data2) #Finance Document

siml_data3 = str(round(float(cosine_similarity(tfidf_matrix_train,tfidf_matrix_test2))*100,2))+'%'
print('similarity between standard document and trail document2 is',siml_data3) #Mask Document


# Word Cloud

for i in list_corpus:
    
    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    bag_of_words = vectorizer.fit_transform(i)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    
    #Generating wordcloud and saving as jpg image
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(15,8))
    plt.title('Most frequently occurring Unigrams')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    

    
    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(i)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    
    #Generating wordcloud and saving as jpg image
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(15,8))
    plt.title('Most frequently occurring bigrams connected by same colour and font size')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    #Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    bag_of_words = vectorizer.fit_transform(i)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    
    #Generating wordcloud and saving as jpg image
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(15,8))
    plt.title('Most frequently occurring trigrams connected by same colour and font size')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


