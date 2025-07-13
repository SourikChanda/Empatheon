import os
import pandas as pd
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# --- CSV is in the root directory ---
CSV_PATH = 'ThousandVoicesOfTrauma.csv'

# --- Step 1: Load and preprocess data ---
def load_and_preprocess_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found at: {path}")
        return [], []
    
    df = pd.read_csv(path)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)
    return df['question'].tolist(), df['response'].tolist()

# --- Step 2: PTSD Chatbot Model ---
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

# --- Step 3: Streamlit UI ---
st.set_page_config(page_title="PTSD Support Chatbot", layout="centered")
st.title("🧠 PTSD Support Chatbot")
st.markdown("This chatbot provides support for PTSD-related concerns. Type your concern below.")

@st.cache_resource
def load_chatbot():
    questions, responses = load_and_preprocess_data(CSV_PATH)
    if not questions:
        return None
    bot = PTSDChatbot()
    bot.train(questions, responses)
    return bot

chatbot = load_chatbot()

if chatbot:
    user_input = st.text_input("You:", "")

    if user_input:
        response = chatbot.get_response(user_input)
        st.markdown(f"**Bot:** {response}")
else:
    st.warning("⚠️ Chatbot could not be loaded. Please check your dataset file location and format.")
