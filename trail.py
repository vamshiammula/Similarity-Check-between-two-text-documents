from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for,abort,send_from_directory
from nltk.sentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename    #helps us to convert bad filename into secure filename
import os

import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
import re
# from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.summarizer import summarize


# ...



#This function is used for preprocessing text summary

def process_sum(row):
     data = row

     #Removes unicode strings like "\u002c" and "x96"
     data = re.sub(r'(\\u[0-9A-Fa-f]+)',r" ", data)
     data = re.sub(r'[^\x00-\x7f]',r" ",data)
     data = re.sub(r'\([^)]*\)', '', data)
     #Remove additional white spaces
     data = re.sub('[\s]+', ' ', data)
     data = re.sub('[\n]+', ' ', data)
     row = data
     return row


# This function is used for Standardizing Data

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

# this function is used for removing punctuation marks, Lemmatizing words and tokenzing

def processRow(row):
    data = row
    # Lower case
    data = data.lower()

    # Removes unicode strings like "\u002c" and "x96"
    data = re.sub(r'(\\u[0-9A-Fa-f]+)', r" ", data)
    data = re.sub(r'[^\x00-\x7f]', r" ", data)
    # convert any url to URL
    data = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', data)
    # Convert any @Username to "AT_USER"
    data = re.sub('@[^\s]+', 'AT_USER', data)
    # Remove additional white spaces
    data = re.sub('[\s]+', ' ', data)
    data = re.sub('[\n]+', ' ', data)
    # Remove not alphanumeric symbols white spaces
    data = re.sub(r'[^\w]', ' ', data)
    # Removes hastag in front of a word """
    data = re.sub(r'#([^\s]+)', r'\1', data)
    # Replace #word with word
    data = re.sub(r'#([^\s]+)', r'\1', data)
    # Remove :( or :)
    data = data.replace(':)', "")
    data = data.replace(':(', "")
    # remove multiple exclamation
    data = re.sub(r"(\!)\1+", ' ', data)
    # remove multiple question marks
    data = re.sub(r"(\?)\1+", ' ', data)
    # remove multistop
    data = re.sub(r"(\.)\1+", ' ', data)
    # lemma
    data = " ".join([Word(word).lemmatize() for word in data.split()])
    # stemmer
    # st = PorterStemmer()
    # data=" ".join([st.stem(word) for word in data.split()])
    # Removes emoticons from text
    data = re.sub(
        ':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:',
        "", data)
    # trim
    data = data.strip('\'"')
    # tokens of words
    data = sent_tokenize(data)
    # removing stop words
    data = [word for word in data if not word in stopwords.words()]
    row = data
    return row

#This function is used for sentiment analysis

def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    print('Positive polarity score:',str(round(float(polarity_scores['pos'])*100,2))+'%')
    print('Negative polarity score:',str(round(float(polarity_scores['neg'])*100,2))+'%')
    if polarity_scores['neg'] > polarity_scores['pos']:
        return 'negative'
    else:
        return 'positive'

# it will create Flask instance

app = Flask(__name__)

# Here i am reading my standard text document

data = open("about_masks.txt","r")
data = data.read()
d = {'WHO':'World Health Organization','COVID-19':'Covid19'}
data = replace_all(data, d)


# with the help processRow function we are doing text preprocessing for our standard text

data = processRow(data)

#spelling correction

correct_words = []
spell = Speller(lang='en')
for i in data:
    k = spell(i)
    correct_words.append(k)

data = pd.DataFrame({'text':correct_words})
with open("stopwords_wo_not.txt","r") as sw:
    stop = sw.read()
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['text']

corpus = data['text'].to_list()

train_set = corpus

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)



app.config['MAX_CONTENT_LENGTH'] = 1024*1024*4 #4MB
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

'''

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    #response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.cache_control.max_age = 0
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
'''

@app.route('/')
def index( ):
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('simple.html',upfiles=files)

@app.route('/predict',methods=['GET','POST'])
def predict():
    uploaded_file = request.files.get('file')                     # we are getting file from FORM
    #print(uploaded_file)
    filename = secure_filename(uploaded_file.filename)            # clean the filename and store it in variable
    #print(filename)
    if filename != '':                                            # if the filename is not empty
        file_ext = os.path.splitext(filename)[1]                  #get the extension from filename covid.txt ['covid','.txt']
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:       #if extension is not valid
            abort(404)                                            #then stop execution else
        path = os.path.join(app.config['UPLOAD_PATH'],filename)   # make os compatible path string
        #print(path)
        uploaded_file.save(path)           # then save the file with original name
        data2 = open(path,encoding='utf-8').read()
        sent = fetch_sentiment_using_SIA(data2)
        text = summarize(data2, ratio=0.05)
        data_sum = process_sum(text)
        data2 = replace_all(data2, d)
        data2 = processRow(data2)
        #print(data2)
        correct_words2 = []

        spell = Speller(lang='en')
        for i in data2:
            k = spell(i)
            correct_words2.append(k)

        data2 = pd.DataFrame({'text': correct_words2})
        with open("stopwords_wo_not.txt", "r") as sw:
            stop = sw.read()
        data2['text'] = data2['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        data2['text']

        corpus2 = data2['text'].to_list()
        tfidf_matrix_test1 = tfidf_vectorizer.transform(corpus2)
        siml_data2 = round(float(cosine_similarity(tfidf_matrix_train, tfidf_matrix_test1)) * 100, 2)
        '''
        vectorizer = CountVectorizer(ngram_range=(1, 1))
        bag_of_words = vectorizer.fit_transform(corpus2)
        vectorizer.vocabulary_
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # Generating wordcloud and saving as jpg image
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 100
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, background_color='white')
        wordCloud.generate_from_frequencies(words_dict)
        plt.figure(figsize=(15, 8))
        plt.title('Most frequently occurring Unigrams')
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis("off")

        wordCloud.to_file("static/unigram.jpg")
        '''
        # Using count vectoriser to view the frequency of bigrams
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        bag_of_words = vectorizer.fit_transform(corpus2)
        vectorizer.vocabulary_
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # Generating wordcloud and saving as jpg image
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 100
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, background_color='black')
        wordCloud.generate_from_frequencies(words_dict)
        plt.figure(figsize=(15, 8))
        plt.title('Most frequently occurring bigrams connected by same colour and font size')
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis("off")

        wordCloud.to_file("static/bigram.jpg")

        # Using count vectoriser to view the frequency of bigrams
        vectorizer = CountVectorizer(ngram_range=(3, 3))
        bag_of_words = vectorizer.fit_transform(corpus2)
        vectorizer.vocabulary_
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # Generating wordcloud and saving as jpg image
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 100
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, background_color='black')
        wordCloud.generate_from_frequencies(words_dict)
        plt.figure(figsize=(15, 8))
        plt.title('Most frequently occurring trigrams connected by same colour and font size')
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis("off")

        wordCloud.to_file("static/trigram.jpg")


        return render_template('result.html', prediction=siml_data2,filename=filename,data_sum = data_sum, sent = sent)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
