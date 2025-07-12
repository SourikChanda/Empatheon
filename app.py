import zipfile
import os
import pandas as pd
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# --- UNZIP dataset if not already done ---
import os
import zipfile

zip_path = 'ThousandVoicesOfTrauma.zip'
extract_to = 'data/'

# Create target folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Unzip to the data folder
if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
 df = pd.read_csv('data/ptsd_dataset.csv')



# --- LOAD and Preprocess Dataset ---
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)
    questions = df['question'].tolist()
    responses = df['response'].tolist()
    return questions, responses

# --- PTSDChatbot Model Class ---
class PTSDChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

# --- STREAMLIT APP ---
@st.cache_resource
def load_chatbot():
    csv_path = os.path.join(extract_to, 'ptsd_dataset.csv')
    questions, responses = load_and_preprocess_data(csv_path)
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
