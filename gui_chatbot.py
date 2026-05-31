# ============================================
# AI LEARNING ASSISTANT CHATBOT GUI
# ============================================

import nltk
import random
import string
import tkinter as tk

from tkinter import *

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# DOWNLOAD NLTK PACKAGES
# ============================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ============================================
# LOAD DATASET
# ============================================

f = open('ai_dataset.txt', 'r', errors='ignore')

raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)

# ============================================
# NLP PREPROCESSING
# ============================================

lemmer = WordNetLemmatizer()

def LemTokens(tokens):

    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict(
    (ord(punct), None) for punct in string.punctuation
)

def LemNormalize(text):

    return LemTokens(
        nltk.word_tokenize(
            text.lower().translate(remove_punct_dict)
        )
    )

# ============================================
# GREETING
# ============================================

GREETING_INPUTS = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening"
)

GREETING_RESPONSES = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Hello, ask me anything about AI."
]

def greeting(sentence):

    for word in sentence.split():

        if word.lower() in GREETING_INPUTS:

            return random.choice(GREETING_RESPONSES)

# ============================================
# CHATBOT RESPONSE
# ============================================

def response(user_response):

    chatbot_response = ''

    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(
        tokenizer=LemNormalize,
        stop_words='english'
    )

    tfidf = TfidfVec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf)

    idx = vals.argsort()[0][-2]

    flat = vals.flatten()

    flat.sort()

    req_tfidf = flat[-2]

    if req_tfidf == 0:

        chatbot_response = (
            "Sorry, I could not understand your question."
        )

        return chatbot_response

    else:

        chatbot_response = chatbot_response + sent_tokens[idx]

        return chatbot_response

# ============================================
# SEND MESSAGE FUNCTION
# ============================================

def send():

    user_message = entry_box.get()

    chat_log.config(state=NORMAL)

    chat_log.insert(END, "You : " + user_message + '\n\n')

    if greeting(user_message) is not None:

        bot_response = greeting(user_message)

    else:

        bot_response = response(user_message)

    chat_log.insert(END, "Bot : " + bot_response + '\n\n')

    chat_log.config(state=DISABLED)

    entry_box.delete(0, END)

# ============================================
# GUI WINDOW
# ============================================

window = Tk()

window.title("AI Learning Assistant Chatbot")

window.geometry("600x700")

window.configure(bg="#1e1e1e")

# ============================================
# CHAT AREA
# ============================================

chat_log = Text(
    window,
    bg="#2b2b2b",
    fg="white",
    font=("Arial", 12),
    width=70,
    height=30
)

chat_log.config(state=DISABLED)

chat_log.pack(padx=10, pady=10)

# ============================================
# ENTRY BOX
# ============================================

entry_box = Entry(
    window,
    bg="#3c3f41",
    fg="white",
    font=("Arial", 14),
    width=40
)

entry_box.pack(pady=10)

# ============================================
# SEND BUTTON
# ============================================

send_button = Button(
    window,
    text="Send",
    command=send,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold")
)

send_button.pack()

# ============================================
# RUN APPLICATION
# ============================================

window.mainloop()