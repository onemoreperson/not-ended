# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
wml = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix




df = pd.read_csv('C:/Users/Оля/Desktop/All-seasons.csv')
a = df.groupby('Character')
Stan = df[df['Character'] == 'Stan']#a.get_group('Stan')
Cartman = df[df['Character'] == 'Kyle']#a.get_group('Cartman')
Kyle = df[df['Character'] == 'Cartman']#a.get_group('Kyle')
Kenny = df[df['Character'] == 'Kenny']#a.get_group('Kenny')
#print(Kyle[:800])
#x = 0
mass = [i for i in Kenny['Line'][:100]]

def lemm(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)    
    mass = [wml.lemmatize(word) for word in word_tokens if word not in stop_words]
    return mass    
#for i in mass:
#    print(lemm(i))

xtrain, xtest, ytrain, ytest = train_test_split(df['Line'], df['Character'], test_size=0.2)#msg label
#print(ytest.value_counts())



train = pd.concat([Stan[:350], Kyle[:350], Cartman[:350], Kenny[:350]], ignore_index = True)
#print(train.groupby('Character').describe())
test = pd.concat([Stan[:350], Kyle[:350], Cartman[:350], Kenny[:350]], ignore_index = True)

x_train = list(train['Line'])
y_train = np.array(train['Character'])
x_test = list(test['Line'])
y_test = np.array(test['Character'])

cv = CountVectorizer(tokenizer = lemm)
cvtrain = cv.fit_transform(x_train)
mnb = MultinomialNB()
mnb.fit(cvtrain, y_train)
testcv = cv.transform(x_test)
y_pred = mnb.predict(testcv)
mat = confusion_matrix(y_test, y_pred)

