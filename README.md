#  Movie Chatbot - NLP with BERT

This project aims to develop a **Telegram Chatbot** based on natural language processing using the **BERT** model. The model has been trained using **movie data from Kaggle** and can provide recommendations based on user preferences via Telegram.

##  Project Overview

In this project, a **BERT-based** model has been trained to analyze movie details and provide recommendations based on user queries.

Dataset used: [Movies Details Dataset](https://www.kaggle.com/datasets/sachinkumar62/movies-details)

##  Technologies Used
- **Python**
- **BERT**
- **Natural Language Processing**
- **Telegram Bot API**
- **Pandas & NumPy**
- **Scikit-Learn**
- **Sentence-Transformers**
- **NLTK**
- **Regex**

##  Dataset
The **Movies Details Dataset** from Kaggle was used to train the model. The dataset includes the following information:
- **Movie Title**
- **Overview**
- **Vote Average**

##  Telegram Bot
Our bot allows users to describe the type and content of the movie they want to watch via Telegram. Based on the provided criteria, the bot lists the most suitable movies and provides recommendations.

##  Installation & Execution
### 1️⃣ Install Required Dependencies
```bash
pip install numpy pandas nltk sentence-transformers scikit-learn regex python-telegram-bot
```

### 2️⃣ Download the Kaggle Dataset
Download the [Movies Details Dataset](https://www.kaggle.com/datasets/sachinkumar62/movies-details) and place it in the project directory.

### 3️⃣ Train the Model
```bash
python bert_vectorizer.py
```

### 4️⃣ Set Up the Telegram Bot Token
To run your Telegram bot, enter the **TOKEN** obtained from **BotFather** in the `telegram_bot.py` file.

### 5️⃣ Start the Telegram Bot
```bash
python telegram_bot.py
```

##  Example Usage
**User:** *"Recommend a science fiction movie featuring time travel and parallel universes."*  
**Bot:** *"See You Yesterday (2019) - Two teenage science prodigies spend their time inventing time machines..."*

##  Contributing
If you would like to contribute to the project, feel free to submit a **pull request** or open an **issue**.
