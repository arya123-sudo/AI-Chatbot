# ============================================
# VOICE AI LEARNING ASSISTANT CHATBOT
# ============================================

import nltk
import random
import string
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
# SPEECH ENGINE
# ============================================

engine = pyttsx3.init()

def speak(text):

    print("Bot :", text)

    engine.say(text)

    engine.runAndWait()

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
# RESPONSE FUNCTION
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
# VOICE INPUT
# ============================================

# ============================================
# IMPROVED VOICE INPUT
# ============================================

recognizer = sr.Recognizer()

def take_voice():

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.adjust_for_ambient_noise(source, duration=1)

        audio = recognizer.listen(
            source,
            timeout=5,
            phrase_time_limit=8
        )

    try:

        text = recognizer.recognize_google(audio)

        print("You :", text)

        return text.lower()

    except sr.UnknownValueError:

        return "none"

    except sr.RequestError:

        return "network error"

# ============================================
# MAIN CHATBOT LOOP
# ============================================

speak("AI Learning Assistant started.")

while True:

    user_response = take_voice()

    # Ignore empty voice
    if user_response == "none":

        print("Could not understand audio")

        continue

    # Internet issue
    if user_response == "network error":

        speak("Internet connection problem.")

        continue

    # Exit commands
    if user_response in [
        "bye",
        "exit",
        "stop",
        "close",
        "end session",
        "quit"
    ]:

        speak("Goodbye. Keep learning AI.")

        break

    # Greeting
    elif greeting(user_response) is not None:

        speak(greeting(user_response))

    # Normal response
    else:

        bot_response = response(user_response)

        speak(bot_response)

# ============================================
# MAIN CHATBOT
# ============================================

speak("AI Learning Assistant started.")

while True:

    user_response = take_voice()

    if user_response == "bye":

        speak("Goodbye. Keep learning AI.")

        break

    elif greeting(user_response) is not None:

        speak(greeting(user_response))

    else:

        bot_response = response(user_response)

        speak(bot_response)