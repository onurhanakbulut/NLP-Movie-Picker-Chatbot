#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:44:11 2025

@author: ohabulut
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sentence_transformers import SentenceTransformer

#------------------preprocess-----------------------
old_data = pd.read_csv('moviesss.csv')

# print(old_data.isnull().sum())
# print(old_data[old_data.isnull().any(axis=1)])
old_data.dropna(subset=['overview'], inplace = True)
# print(old_data.isnull().sum())



data = old_data.iloc[:,2:4]
imdb = old_data.iloc[:,6:7]

data = data.reset_index(drop=True)

for i in range (len(data)):
    
    data.loc[i, 'title'] = re.sub(r"[^\w\s]", " ", data.loc[i, "title"]).lower().strip()


titles = list(data["title"])


overviews = []
for i in range (len(data)):
    overview = re.sub(r'[^\w\s]', ' ',data['overview'] [i])
    overview =overview.lower()
    overview = overview.split()
    overview = [ps.stem(words) for words in overview if not words in set(stopwords.words('english'))]
    overview = ' '.join(overview)
    overviews.append(overview)

#--------------BERT--------------------------
model = SentenceTransformer('all-mpnet-base-v2')


bert_vectors = model.encode(overviews, convert_to_tensor=True)

bert_vectors_np = bert_vectors.cpu().numpy()

bert_data = pd.DataFrame({'title': titles, 'vector': list(bert_vectors_np)})

bert_data.to_pickle('bert_movie_vectors.pkl')



















