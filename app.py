import zipfile
import os

zip_path = 'ThousandVoicesOfTrauma.zip'
extract_to = 'ptsd_dataset.csv'

if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
import pandas as pd

df = pd.read_csv('data/ptsd_dataset.csv')

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PTSDChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight yet effective
        self.index = None
        self.questions = []
        self.answers = []

    def train(self, questions, answers):
        self.questions = questions
        self.answers = answers
        embeddings = self.model.encode(questions, convert_to_tensor=False)
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings).astype('float32'))

    def get_response(self, user_input):
        query_embedding = self.model.encode([user_input])[0].astype('float32')
        D, I = self.index.search(np.array([query_embedding]), k=1)
        return self.answers[I[0][0]]
import pandas as pd
from sklearn.utils import shuffle

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)

    # Data augmentation: simple paraphrasing (optional)
    # You could add paraphrased versions here manually or use a paraphrasing model

    questions = df['question'].tolist()
    responses = df['response'].tolist()
    return questions, responses
import streamlit as st
from model.chatbot_model import PTSDChatbot
from utils.preprocess import load_and_preprocess_data

@st.cache_resource
def load_chatbot():
    questions, responses = load_and_preprocess_data('data/ptsd_dataset.csv')
    bot = PTSDChatbot()
    bot.train(questions, responses)
    return bot

st.set_page_config(page_title="PTSD Support Chatbot", layout="centered")
st.title("🧠 PTSD Support Chatbot")
st.markdown("This chatbot provides support for PTSD-related concerns. Type your concern below.")

chatbot = load_chatbot()
user_input = st.text_input("You:", "")

if user_input:
    response = chatbot.get_response(user_input)
    st.markdown(f"**Bot:** {response}")
