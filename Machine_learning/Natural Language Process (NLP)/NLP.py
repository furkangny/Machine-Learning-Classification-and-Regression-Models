# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:44:08 2024

@author: frkng
"""
import pandas as pd

#%% import twitter data
data = pd.read_csv(r"gender_classifier.csv", encoding="latin1")
data = pd.concat([data.gender,data.description],axis = 1)
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% cleaning data
# regular expression RE-> for example "[^a-zA-Z]"
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]", " ",first_description) #a-z arasında olmayan ifadeleri bulup boşluk ile değiştirir
description = description.lower() # büyük harften küçük harfe dönüşüm 

#%% stopwords (irrelavent words) -> gereksiz kelimeler

import nltk # natural language tool kit
nltk.download("stopwords") # corpus diye bir klasör indiriliyor
from nltk.corpus import stopwords # sonra corpus klasöründen import ediyoruz

# description = nltk.word_tokenize() yerine -> description.split() <- şeklinde bir kullanım da yapabiliriz.
description = nltk.word_tokenize(description) # ATTENTİON: This use is recommended.

#%% remove unnecessary words 
description = [word for word in description if not word in set(stopwords.words("english"))]

#%% Kelime köklerini buluyoruz
# Lemmatization ----------- loved -> love , gitmeyeceğim -> git

import nltk as nlp

lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)

#%%
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]", " ",description) #a-z arasında olmayan ifadeleri bulup boşluk ile değiştirir
    description = description.lower() # büyük harften küçük harfe dönüşüm 
    description = nltk.word_tokenize(description) # split gibi ama daha iyi
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    
#%% bag of words

from sklearn.feature_extraction.text import CountVectorizer # for bag of words create
max_features = 1000

count_vectorizer = CountVectorizer(max_features=max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("en sık kullanılan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))

#%% 
y = data.iloc[:,0].values # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state= 42)

#%% naive bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
    
# prediction
y_pred = nb.predict(x_test)
print("accuracy: ", nb.score(y_pred.reshape(-1,1), y_test))



















 