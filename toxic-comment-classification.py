#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import  matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import keras
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score , accuracy_score , confusion_matrix , f1_score
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


train_data  =  pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')


# In[3]:


test_data  =  pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')


# In[4]:


test_target =  pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')


# In[5]:


train_data.head()


# In[6]:


test_data.head()


# In[7]:


test_target.head()


# In[8]:


len(test_data)


# In[9]:


len(train_data)


# In[10]:


len(test_target)


# In[11]:


train_data.isnull().sum()


# In[12]:


train_data.describe()


# In[13]:


comments = train_data.drop(['id','comment_text'],axis = 1)
for i in comments.columns :
    print("Percent of {0}s: ".format(i), round(100*comments[i].mean(),2), "%")


# In[14]:


classes = {}
for i in list(comments.columns):
    classes[i] =  comments[i].sum()
n_classes = [classes[i] for i in list(classes.keys())]
classes = list(classes.keys())


# In[15]:


color = ['red','blue','green','yellow','black','orange']
plt.figure(figsize=(12,12))
fig, ax = plt.subplots()
ax.bar(classes,n_classes,color = color)


# In[16]:


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


# In[17]:



train_data.comment_text = train_data.comment_text.apply(clean_text)


# In[18]:


train_data.head()


# In[19]:


nltk.download('stopwords')
sn = SnowballStemmer(language='english')


def stemmer(text):
    words =  text.split()
    train = [sn.stem(word) for word in words if not word in set(stopwords.words('english'))]
    return ' '.join(train)


# In[20]:


train_data.comment_text = train_data.comment_text.apply(stemmer)


# In[21]:


train_data.comment_text.head()


# In[22]:


wordcloud = WordCloud(stopwords=stopwords.words('english'),max_words=50).generate(str(train_data.comment_text))
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[23]:


x =  train_data.comment_text
y =  train_data.drop(['id','comment_text'],axis = 1)


# In[24]:


print(type(x))


# In[25]:


x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.2,random_state = 45)


# In[26]:


x_train


# In[27]:


x_train2 = x_train.to_numpy()
x_test2 =  x_test.to_numpy()


# In[28]:


y_train2 =  y_train.to_numpy()
y_test =  y_test.to_numpy()


# In[29]:


word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',     
    analyzer='word',            
    token_pattern=r'\w{1,}',    
    ngram_range=(1, 3),         
    stop_words='english',
    sublinear_tf=True)

word_vectorizer.fit(x_train2)    
train_word_features = word_vectorizer.transform(x_train2)


# In[30]:


import joblib

joblib.dump(word_vectorizer, open('vectroize2_jlib', 'wb'))

vectorizer = joblib.load('vectroize2_jlib')


# In[31]:


X_train_transformed = vectorizer.transform(x_train2)
X_test_transformed = vectorizer.transform(x_test2)


# In[32]:


print(X_train_transformed)


# In[33]:


log_reg = LogisticRegression(C = 10, penalty='l2', solver = 'liblinear', random_state=45)

classifier = OneVsRestClassifier(log_reg)
classifier.fit(X_train_transformed, y_train)


y_train_pred_proba = classifier.predict_proba(X_train_transformed)
y_test_pred_proba = classifier.predict_proba(X_test_transformed)


roc_auc_score_train = roc_auc_score(y_train2, y_train_pred_proba,average='weighted')
roc_auc_score_test = roc_auc_score(y_test, y_test_pred_proba,average='weighted')

print("ROC AUC Score Train:", roc_auc_score_train)
print("ROC AUC Score Test:", roc_auc_score_test)


# In[34]:


joblib.dump(classifier, open('classifier2_jlib', 'wb'))

word_vectorizer = joblib.load('classifier2_jlib')


# In[35]:


def make_test_predictions(df,classifier):
    df.comment_text = df.comment_text.apply(clean_text)
    df.comment_text = df.comment_text.apply(stemmer)
    X_test = df.comment_text
    X_test =  X_test.to_numpy()
    X_test_transformed = vectorizer.transform(X_test)
    y_test_pred = classifier.predict_proba(X_test_transformed)
    return y_test_pred
    #y_test_pred_df = pd.DataFrame(y_test_pred,columns=comments.columns) 
    #submission_df = pd.concat([df.id, y_test_pred_df], axis=1)
    #submission_df.to_csv('submission.csv', index = False)
    


# In[36]:


xx ={'id':[565],'comment_text':['Shut up your mouth bitch']}
xx = pd.DataFrame(xx)


# In[37]:


#test 1 
make_test_predictions(xx,classifier)


# In[38]:


xx ={'id':[565],'comment_text':['hi I am happy to be here']}
xx = pd.DataFrame(xx)


# In[39]:


#test 2
make_test_predictions(xx,classifier)


# In[40]:


def make_test_predictions(df,classifier):
    df.comment_text = df.comment_text.apply(clean_text)
    df.comment_text = df.comment_text.apply(stemmer)
    X_test = df.comment_text
    X_test =  X_test.to_numpy()
    X_test_transformed = vectorizer.transform(X_test)
    y_test_pred = classifier.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0])
    if result >=1 :
       return("Offensive Comment")
    else :
       return ("Normal Comment")


# In[41]:


comment_text = "fuck you"
comment ={'id':[565],'comment_text':[comment_text]}
comment = pd.DataFrame(comment)
result = make_test_predictions(comment,classifier)
print(result)


# In[42]:


comment_text = "thanks for your help"
comment ={'id':[565],'comment_text':[comment_text]}
comment = pd.DataFrame(comment)
result = make_test_predictions(comment,classifier)
print(result)


# In[43]:


comment_text = "I want to kill you"
comment ={'id':[565],'comment_text':[comment_text]}
comment = pd.DataFrame(comment)
result = make_test_predictions(comment,classifier)
print(result)


# In[44]:


comment_text = "I really like it"
comment ={'id':[565],'comment_text':[comment_text]}
comment = pd.DataFrame(comment)
result = make_test_predictions(comment,classifier)
print(result)


# In[ ]:





# In[ ]:




