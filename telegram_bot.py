#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:21:23 2025

@author: ohabulut
"""
import nest_asyncio
nest_asyncio.apply()


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext





TOKEN = '****************************************'

model = SentenceTransformer("all-mpnet-base-v2")  

bert_data = pd.read_pickle('bert_movie_vectors.pkl')

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def similar_movie(query):
    query_cleaned = " ".join([ps.stem(word) for word in re.sub(r"[^\w\s]", " ", query).lower().split() if word not in stop_words])
    query_vector = model.encode([query_cleaned], convert_to_tensor=True).cpu().numpy()
    
    similarities = cosine_similarity(query_vector, np.vstack(bert_data['vector']))
    best_match = np.argmax(similarities)
    return bert_data.iloc[best_match]['title']




async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("🎬 Welcome to Movie Recommender Bot! Send me a movie description and I'll recommend a similar one.")

async def recommend(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text  # Kullanıcının mesajını al
    movie_title = similar_movie(user_text)  # En uygun filmi bul
    response = f"🎥 I recommend: *{movie_title}*"
    await update.message.reply_text(response, parse_mode="Markdown")





# Telegram Botu Başlat
app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))  # /start komutu için
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, recommend))  # Kullanıcının mesajını işle

print("🚀 Bot Başlatılıyor...")
app.run_polling()






#--------maunuel query--------------


# query = 'an action movie with superheroes'
# matched_movie = similar_movie(query)
# print(matched_movie)







