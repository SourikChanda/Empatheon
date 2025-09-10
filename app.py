import os
import zipfile
import pandas as pd
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# --- Paths ---
zip_path = 'thousand_voices_of_trauma.zip'
extract_to = 'data/'
csv_path = os.path.join(extract_to, 'dataset.csv')
# --- CSV is in the root directory ---
CSV_PATH = 'thousand_voices_of_trauma.csv'

os.makedirs(extract_to, exist_ok=True)

# --- Step 1: Extract ZIP and create CSV if needed ---
def extract_and_prepare_data():
    if not os.path.exists(csv_path):
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_contents = zip_ref.namelist()
                st.write("🔍 ZIP Contents:", zip_contents)
                zip_ref.extractall(extract_to)

            # Check if CSV exists now
            extracted_csv = [f for f in os.listdir(extract_to) if f.endswith('.csv')]
            if extracted_csv:
                # Rename first CSV found to dataset.csv
                os.rename(os.path.join(extract_to, extracted_csv[0]), csv_path)
                st.success(f"CSV found and renamed to {csv_path}")
            else:
                # Try to create CSV from questions.txt and responses.txt
                questions_txt = os.path.join(extract_to, 'questions.txt')
                responses_txt = os.path.join(extract_to, 'responses.txt')

                if os.path.exists(questions_txt) and os.path.exists(responses_txt):
                    with open(questions_txt, 'r', encoding='utf-8') as qf, open(responses_txt, 'r', encoding='utf-8') as rf:
                        questions = qf.read().splitlines()
                        responses = rf.read().splitlines()

                    if len(questions) != len(responses):
                        st.error("❌ Number of questions and responses do not match!")
                        return False

                    df = pd.DataFrame({'question': questions, 'response': responses})
                    df.to_csv(csv_path, index=False)
                    st.success(f"Created CSV dataset at {csv_path}")
                else:
                    st.error("❌ No CSV file or questions.txt & responses.txt found in ZIP.")
                    return False
        else:
            st.error("❌ Provided file is not a valid ZIP.")
            return False
    else:
        st.info(f"Using existing dataset at {csv_path}")

    return True

# --- Step 2: Load and preprocess data ---
# --- Step 1: Load and preprocess data ---
def load_and_preprocess_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found at: {path}")
        return [], []
    
    df = pd.read_csv(path)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)
    return df['question'].tolist(), df['response'].tolist()

# --- Step 3: PTSD chatbot model ---
# --- Step 2: PTSD Chatbot Model ---
class PTSDChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_response(self, user_input):
        query_embedding = self.model.encode([user_input], convert_to_numpy=True)
        D, I = self.index.search(np.array(query_embedding), k=1)
        return self.answers[I[0][0]]

# --- Step 4: Streamlit UI ---
# --- Step 3: Streamlit UI ---
st.set_page_config(page_title="PTSD Support Chatbot", layout="centered")
st.title("🧠 PTSD Support Chatbot")
st.markdown("This chatbot provides support for PTSD-related concerns. Type your concern below.")

if extract_and_prepare_data():
    @st.cache_resource
    def load_chatbot():
        questions, responses = load_and_preprocess_data(csv_path)
        bot = PTSDChatbot()
        bot.train(questions, responses)
        return bot
@st.cache_resource
def load_chatbot():
    questions, responses = load_and_preprocess_data(CSV_PATH)
    if not questions:
        return None
    bot = PTSDChatbot()
    bot.train(questions, responses)
    return bot

# Load chatbot
chatbot = load_chatbot()

if chatbot:
    user_input = st.text_input("You:", "")

    if user_input:
        response = chatbot.get_response(user_input)
        st.markdown(f"**Bot:** {response}")
else:
    st.error("Cannot proceed without a valid dataset.")
    st.warning("⚠️ Chatbot could not be loaded. Please check your dataset file location and format.")

