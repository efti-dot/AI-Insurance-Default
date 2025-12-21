import os
import numpy as np
import openai
from dotenv import load_dotenv
from utils import extract_text_from_pdf, split_text, create_embeddings
from prompt import OpenAIConfig
from vectordb import VectorStore
from doc_ai import DocAI
import tempfile
import streamlit as st

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ai = OpenAIConfig(api_key=api_key)

# vector stores define
default_vector_store = VectorStore(dim=1536)   # for initial PDF uploader
doc_ai = DocAI(dim=1536)

def init_session_vector_store():
    global session_vector_store
    if session_vector_store is None:
        session_vector_store = VectorStore(dim=1536)


# for RAG
def process_pdf(file):
    text = extract_text_from_pdf(file)
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)

    texts = [chunk["text"] for chunk in embeddings]
    vectors = [chunk["embedding"] for chunk in embeddings]

    default_vector_store.add(vectors, texts)
    return embeddings



# for user attachments
def process_attachment(file):
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    msg = doc_ai.upload(tmp_path)   # handles text docs into FAISS, images into kb
    return msg





def query(user_input, history):
    # Embed query
    response = openai.embeddings.create(model="text-embedding-3-small", input=user_input)
    user_embedding = response.data[0].embedding

    results = []

    # Search default FAISS
    if default_vector_store.index.ntotal > 0:
        results += default_vector_store.search(user_embedding, top_k=3)

    # Search session FAISS (inside doc_ai)
    if doc_ai.index.ntotal > 0:
        query_vec = np.array([user_embedding]).astype("float32")
        distances, indices = doc_ai.index.search(query_vec, 3)
        results += [doc_ai.metadata[i] for i in indices[0]]

    # Add image descriptions from kb
    if doc_ai.kb:
        results += [f"{d['name']}: {d['content']}" for d in doc_ai.kb]

    
    relevant_text = "\n\n".join(results)

    prompt = f"""You are a warm, friendly Swedish insurance assistant. Use the following context (policies + user attachments):

{relevant_text}

User question:
{user_input}

Guidelines:
- If a user message is NOT related to insurance or insurance assistance, politely decline and guide them back to an insurance-related topic.
- Be supportive and human not robotic.
- Prefer short, actionable guidance; expand only when the user asks for more.
- Respect user's pacing — wait for their response before advancing the flow.
            """
    return ai.get_response(prompt, history)
