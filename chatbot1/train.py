
import json
import pickle
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_sm")

with open("intents.json") as f:

    data = json.load(f)

sentences = []
labels = []
classes = []

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(preprocess(pattern))
        labels.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
y = np.array(labels)

model = LogisticRegression()
model.fit(X, y)

with open("chatbot.pkl", "wb") as f:
    pickle.dump((model, vectorizer, classes), f)

print("✅ Model trained and saved as chatbot.pkl")
