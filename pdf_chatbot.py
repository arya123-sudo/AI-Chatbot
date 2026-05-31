# ============================================
# PDF AI LEARNING ASSISTANT CHATBOT
# ============================================

import nltk
import random
import string
import PyPDF2

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
# READ PDF FILE
# ============================================

pdf_file = open('ai_notes.pdf', 'rb')

pdf_reader = PyPDF2.PdfReader(pdf_file)

raw_text = ""

for page in pdf_reader.pages:

    raw_text += page.extract_text()

raw_text = raw_text.lower()

sent_tokens = nltk.sent_tokenize(raw_text)

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
    "hey"
)

GREETING_RESPONSES = [
    "Hello!",
    "Hi there!",
    "Ask me anything from your AI notes."
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
            "Sorry, I could not find the answer in the PDF."
        )

        return chatbot_response

    else:

        chatbot_response = chatbot_response + sent_tokens[idx]

        return chatbot_response

# ============================================
# CHATBOT START
# ============================================

print("======================================")
print(" PDF AI LEARNING ASSISTANT CHATBOT")
print("======================================")
print("Ask questions from your PDF notes")
print("Type 'bye' to exit")
print()

while True:

    user_response = input("You : ")

    user_response = user_response.lower()

    if user_response != 'bye':

        if greeting(user_response) is not None:

            print("Bot :", greeting(user_response))

        else:

            print("Bot :", response(user_response))

    else:

        print("Bot : Goodbye!")

        break