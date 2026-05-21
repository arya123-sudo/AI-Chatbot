# ============================================
# AI CHATBOT FOR TEXT INFORMATION
# Experiment 10
# ============================================

# Import Libraries
import nltk
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK package
nltk.download('punkt')
nltk.download('punkt_tab')

# ============================================
# READ DATASET FILE
# ============================================

# Open the text file
f = open('data.txt', 'r', errors='ignore')

# Read file and convert to lowercase
raw = f.read().lower()

# Convert text into sentence tokens
sent_tokens = nltk.sent_tokenize(raw)

# ============================================
# GREETING FUNCTION
# ============================================

GREETING_INPUTS = ("hello", "hi", "hey", "good morning", "good evening")

GREETING_RESPONSES = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Welcome!",
    "Hi, how can I help you?"
]

def greeting(sentence):

    for word in sentence.split():

        if word.lower() in GREETING_INPUTS:

            return random.choice(GREETING_RESPONSES)

# ============================================
# CHATBOT RESPONSE FUNCTION
# ============================================

def response(user_response):

    chatbot_response = ''

    # Add user response to sentence list
    sent_tokens.append(user_response)

    # Convert text into TF-IDF vectors
    TfidfVec = TfidfVectorizer()

    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Find similarity
    vals = cosine_similarity(tfidf[-1], tfidf)

    idx = vals.argsort()[0][-2]

    flat = vals.flatten()

    flat.sort()

    req_tfidf = flat[-2]

    # If no similarity found
    if req_tfidf == 0:

        chatbot_response = "Sorry, I don't understand your question."

        return chatbot_response

    else:

        chatbot_response = chatbot_response + sent_tokens[idx]

        return chatbot_response

# ============================================
# START CHATBOT
# ============================================

print("==========================================")
print("      AI CHATBOT FOR TEXT INFORMATION")
print("==========================================")
print("Type 'bye' to exit the chatbot")
print()

while True:

    user_response = input("You : ")

    user_response = user_response.lower()

    # Exit Condition
    if user_response != 'bye':

        if user_response in ['thanks', 'thank you']:

            print("Chatbot : You are welcome!")

            break

        else:

            # Greeting response
            if greeting(user_response) is not None:

                print("Chatbot :", greeting(user_response))

            else:

                print("Chatbot :", response(user_response))

    else:

        print("Chatbot : Goodbye! Have a nice day.")

        break