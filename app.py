import os
import zipfile
import pandas as pd
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# --- Paths ---
ZIP_PATH = 'thousand_voices_of_trauma.zip'  # optional
EXTRACT_TO = 'data/'
CSV_PATH = 'thousand_voices_of_trauma.csv'  # main dataset

os.makedirs(EXTRACT_TO, exist_ok=True)

# --- Step 1: Extract ZIP if needed ---
def extract_and_prepare_data():
    """Extracts dataset if ZIP is present, otherwise checks CSV exists."""
    if not os.path.exists(CSV_PATH):
        if zipfile.is_zipfile(ZIP_PATH):
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                st.write("🔍 Extracting ZIP contents...")
                zip_ref.extractall(EXTRACT_TO)

            # Look for CSV
            extracted_csv = [f for f in os.listdir(EXTRACT_TO) if f.endswith('.csv')]
            if extracted_csv:
                os.rename(os.path.join(EXTRACT_TO, extracted_csv[0]), CSV_PATH)
                st.success(f"✅ Dataset extracted to {CSV_PATH}")
            else:
                st.error("❌ No CSV file found inside ZIP.")
                return False
        else:
            st.error(f"❌ Dataset not found at {CSV_PATH} or {ZIP_PATH}")
            return False
    else:
        st.info(f"ℹ️ Using existing dataset at {CSV_PATH}")
    return True

# --- Step 2: Load and preprocess data ---
def load_and_preprocess_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found at: {path}")
        return [], []
    
    df = pd.read_csv(path).dropna()
    df = shuffle(df).reset_index(drop=True)
    return df['question'].tolist(), df['response'].tolist()

# --- Step 3: PTSD chatbot model ---
class PTSDChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.answers = []

    def train(self, questions, responses):
        """Build FAISS index for retrieval."""
        if not questions:
            return False
        embeddings = self.model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.answers = responses
        return True

    def get_response(self, user_input, k=1):
        """Return closest response from dataset."""
        if self.index is None:
            return "⚠️ Chatbot is not trained."
        query_embedding = self.model.encode([user_input], convert_to_numpy=True)
        D, I = self.index.search(np.array(query_embedding), k=k)
        return self.answers[I[0][0]]

# --- Step 4: Streamlit UI ---
st.set_page_config(page_title="PTSD Support Chatbot", layout="centered")
st.title("🧠 PTSD Support Chatbot")
st.markdown("This chatbot provides support for PTSD-related concerns. "
            "⚠️ **Disclaimer:** This is not a diagnostic tool. If you’re struggling, please seek professional help.")

if extract_and_prepare_data():
    @st.cache_resource
    def load_chatbot():
        questions, responses = load_and_preprocess_data(CSV_PATH)
        bot = PTSDChatbot()
        if bot.train(questions, responses):
            return bot
        return None

    chatbot = load_chatbot()

    if chatbot:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Type your concern..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = chatbot.get_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.error("⚠️ Chatbot could not be trained. Check dataset format.")
