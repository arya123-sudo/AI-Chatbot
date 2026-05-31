# ============================================
# FINAL AI LEARNING ASSISTANT
# ALL FEATURES IN ONE FILE
# ============================================

import streamlit as st
import json
import nltk
import random
import string
import PyPDF2
import speech_recognition as sr
import pyttsx3

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

with open("ai_dataset.txt", "r", errors="ignore") as f:

    raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)

conversation_history = []

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
    "good evening",
    "hii"
)

GREETING_RESPONSES = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Welcome!",
    "Hi, ask me anything about AI."
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
        "define ai",
        "artificial intelligence"
    ],

    "machine_learning": [
        "what is machine learning",
        "define machine learning",
        "ml"
    ],

    "deep_learning": [
        "what is deep learning",
        "define deep learning"
    ],

    "nlp": [
        "what is nlp",
        "natural language processing"
    ],

    "python": [
        "what is python",
        "python language"
    ],

    "help": [
        "help",
        "can you help me",
        "i need help",
        "how can you help"
    ],

    "bye": [
        "bye",
        "goodbye",
        "exit"
    ],

    "thanks": [
        "thanks",
        "thank you"
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
        "Natural Language Processing enables computers to understand human language.",

    "python":
        "Python is a popular programming language used in AI and Machine Learning.",

    "help":
        "Yes, I can help you with AI, Machine Learning, NLP, Python, and Data Science topics.",

    "bye":
        "Goodbye! Keep learning AI.",

    "thanks":
        "You are welcome!"
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
# SMART RESPONSE FUNCTION
# ============================================

def response(user_response):

    chatbot_response = ""

    conversation_history.append(user_response)

    context_text = " ".join(conversation_history[-3:])

    sent_tokens.append(context_text)

    TfidfVec = TfidfVectorizer(
        tokenizer=LemNormalize,
        stop_words="english"
    )

    tfidf = TfidfVec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf)

    idx = vals.argsort()[0][-2]

    flat = vals.flatten()

    flat.sort()

    req_tfidf = flat[-2]

    # ============================================
    # CONFIDENCE CHECK
    # ============================================

    if req_tfidf < 0.2:

        fallback_responses = [

            "Sorry, I could not understand that properly.",

            "Please ask questions related to AI, ML, NLP, or Python.",

            "I am an AI Learning Assistant. Try asking AI-related questions.",

            "Can you rephrase your question?"
        ]

        return random.choice(fallback_responses)

    else:

        chatbot_response = chatbot_response + sent_tokens[idx]

        return chatbot_response
    
    # ============================================
# SAVE CHAT HISTORY
# ============================================

def save_chat(user, bot):

    chat_data = {

        "user": user,
        "bot": bot
    }

    try:

        with open("chat_history.json", "r") as file:

            data = json.load(file)

    except:

        data = []

    data.append(chat_data)

    with open("chat_history.json", "w") as file:

        json.dump(data, file, indent=4)

# ============================================
# VOICE ASSISTANT
# ============================================

recognizer = sr.Recognizer()

engine = pyttsx3.init()

# ============================================
# SPEAK FUNCTION
# ============================================

def speak(text):

    try:

        engine.stop()

        engine.say(text)

        engine.runAndWait()

    except RuntimeError:

        pass
def take_voice():

    with sr.Microphone() as source:

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)

        return text.lower()

    except:

        return "Could not recognize voice."

# ============================================
# PDF READER
# ============================================

def read_pdf(uploaded_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted

    return text

# ============================================
# STREAMLIT PAGE
# ============================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align:center;'>
    🤖 AI Learning Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Ask anything about AI, ML, NLP, Python, or Data Science."
)

# ============================================
# CHAT HISTORY
# ============================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ============================================
# DISPLAY PREVIOUS MESSAGES
# ============================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ============================================
# USER INPUT
# ============================================

prompt = st.chat_input("Ask your question...")

if prompt:

    # USER MESSAGE
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # ============================================
    # BOT RESPONSE
    # ============================================

    if greeting(prompt) is not None:

        bot_response = greeting(prompt)

    else:

        intent = detect_intent(prompt)

        if intent is not None:

            bot_response = intent_responses[intent]

        else:

            bot_response = response(prompt)

    # BOT MESSAGE
    with st.chat_message("assistant"):

        st.markdown(bot_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_response
        }
    )
    save_chat(prompt, bot_response)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("⚡ Features")

# ============================================
# VOICE ASSISTANT
# ============================================

if st.sidebar.button("🎤 Start Voice Assistant"):

    voice_text = take_voice()

    st.sidebar.success("You Said: " + voice_text)

    if greeting(voice_text) is not None:

        voice_response = greeting(voice_text)

    else:

        intent = detect_intent(voice_text)

        if intent is not None:

            voice_response = intent_responses[intent]

        else:

            voice_response = response(voice_text)

    st.sidebar.info("Bot: " + voice_response)

    speak(voice_response)

# ============================================
# PDF UPLOAD
# ============================================

st.sidebar.subheader("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload AI Notes",
    type=["pdf"]
)

if uploaded_file is not None:

    pdf_text = read_pdf(uploaded_file)

    st.sidebar.success("PDF Loaded Successfully")

    st.sidebar.text_area(
        "PDF Preview",
        pdf_text[:2000],
        height=250
    )

# ============================================
# ABOUT
# ============================================

st.sidebar.subheader("ℹ️ About")

st.sidebar.info(
    """
    AI Learning Assistant Chatbot

    Features:
    ✔ NLP Chatbot
    ✔ Intent Detection
    ✔ Conversation Memory
    ✔ Voice Assistant
    ✔ PDF Reader
    ✔ Streamlit UI
    """
)