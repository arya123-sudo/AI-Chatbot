# ============================================
# STREAMLIT AI LEARNING ASSISTANT CHATBOT
# ============================================

import streamlit as st
import nltk
import random
import string

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# DOWNLOAD NLTK DATA
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

conversation_history = []

# ============================================
# TEXT PREPROCESSING
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
    "hey"
)

GREETING_RESPONSES = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Ask me anything about AI."
]

def greeting(sentence):

    for word in sentence.split():

        if word.lower() in GREETING_INPUTS:

            return random.choice(GREETING_RESPONSES)

# ============================================
# INTENT PATTERNS
# ============================================

intent_patterns = {

    "ai_definition": [
        "what is ai",
        "define ai"
    ],

    "machine_learning": [
        "what is machine learning",
        "define machine learning"
    ],

    "deep_learning": [
        "what is deep learning"
    ],

    "nlp": [
        "what is nlp"
    ]
}

# ============================================
# INTENT RESPONSES
# ============================================

intent_responses = {

    "ai_definition":
        "Artificial Intelligence is a branch of computer science that enables machines to simulate human intelligence.",

    "machine_learning":
        "Machine Learning is a subset of AI that enables systems to learn automatically from data.",

    "deep_learning":
        "Deep Learning uses neural networks with multiple hidden layers.",

    "nlp":
        "Natural Language Processing enables computers to understand human language."
}

# ============================================
# DETECT INTENT
# ============================================

def detect_intent(user_input):

    user_input = user_input.lower()

    for intent, patterns in intent_patterns.items():

        for pattern in patterns:

            if pattern in user_input:

                return intent

    return None

# ============================================
# RESPONSE FUNCTION
# ============================================

def response(user_response):

    chatbot_response = ''

    conversation_history.append(user_response)

    context_text = " ".join(conversation_history[-3:])

    sent_tokens.append(context_text)

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
            "Sorry, I could not understand."
        )

        return chatbot_response

    else:

        chatbot_response = chatbot_response + sent_tokens[idx]

        return chatbot_response

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Learning Assistant Chatbot")

st.write("Ask me anything about AI, ML, NLP, Python")

user_input = st.text_input("Enter your question")

if st.button("Send"):

    if greeting(user_input) is not None:

        st.success(greeting(user_input))

    else:

        intent = detect_intent(user_input)

        if intent is not None:

            st.success(intent_responses[intent])

        else:

            st.success(response(user_input))