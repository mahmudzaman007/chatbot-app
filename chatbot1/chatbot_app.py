
import json
import random
import pickle
import spacy
import streamlit as st
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
model, vectorizer, classes = pickle.load(open("chatbot.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

if "context" not in st.session_state:
    st.session_state.context = None

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

def predict_class(text):
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    return model.predict(X)[0]

def get_intent_data(intent_tag):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return intent
    return None

def get_response(intent_tag):
    intent = get_intent_data(intent_tag)
    if "context_filter" in intent and st.session_state.context != intent["context_filter"]:
        return "Can you clarify that a bit?"
    if "context_set" in intent:
        st.session_state.context = intent["context_set"]
    return random.choice(intent["responses"])

st.title("🤖 Contextual Chatbot")
st.markdown("Type a message below to talk with the bot.")

user_input = st.text_input("You:", "")

if user_input:
    intent = predict_class(user_input)
    response = get_response(intent)
    st.text_area("Bot:", value=response, height=100)
