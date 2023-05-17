
from turtle import width
from matplotlib.ft2font import BOLD
import numpy as np
import pickle
import seaborn as sns
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import pandas as pd 
import  matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import keras
import re
import time
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import pickle
from PIL import Image

import base64
from sklearn.feature_extraction.text import TfidfVectorizer
fig = plt.figure(figsize=(12, 5))


#loading_model = pickle.load(open('classifier2_jlib', 'rb'))
loading_model=joblib.load('classifier2_jlib')
vectorizer = joblib.load('vectroize2_jlib')


#st.image("stroke\logo.png")


def  clean_text(text):
    text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    
    return text

nltk.download('stopwords')
sn = SnowballStemmer(language='english')



def stemmer(text):
    words =  text.split()
    train = [sn.stem(word) for word in words if not word in set(stopwords.words('english'))]
    return ' '.join(train)

def make_test_predictions(df,classifier):
    df.comment_text = df.comment_text.apply(clean_text)
    df.comment_text = df.comment_text.apply(stemmer)
    X_test = df.comment_text
    X_test =  X_test.to_numpy()
    X_test_transformed = vectorizer.transform(X_test)
    y_test_pred = classifier.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0]) 
    l=y_test_pred[0]
    new_data=l.tolist()
    name=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    chart=pd.DataFrame({'x': name,'Comments Categories': l})
    chart = chart.rename(columns={'x':'index'}).set_index('index')
    #Line Chart
    st.line_chart(np.round(chart,2))
    
    if result >=1.5 :
       return("Offensive Comment")
    else :
       return ("Normal Comment")

def main():
    st.title('toxic-comment-classification')

    
    comment_text = st.text_area("comment_text")
    comment ={'id':[565],'comment_text':[comment_text]}
    comment = pd.DataFrame(comment)
    

    Diagnosis = ''
    if st.button("Prediction Result"):
        Diagnosis = make_test_predictions(comment,loading_model)

    st.success(Diagnosis)
   

if __name__ == '__main__':
    main()

    

