import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral
from transformers import pipeline
from sklearn.preprocessing import normalize

# mistral API key
api_key = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"
os.environ["MISTRAL_API_KEY"] = api_key

client = Mistral(api_key=api_key)

# intent classification model (Zero-shot using Hugging Face)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# policies
policies = {
    "Academic Annual Leave": "https://www.udst.edu.qa/.../academic-annual-leave-policy",
    "Academic Appraisal": "https://www.udst.edu.qa/.../academic-appraisal-policy",
    "Intellectual Property": "https://www.udst.edu.qa/.../intellectual-property-policy",
    "Credit Hour": "https://www.udst.edu.qa/.../credit-hour-policy",
    "Research Ethics": "https://www.udst.edu.qa/.../research-ethics-policy",
    "Student Conduct": "https://www.udst.edu.qa/.../student-conduct-policy",
    "Faculty Hiring": "https://www.udst.edu.qa/.../faculty-hiring-policy",
    "Examination Rules": "https://www.udst.edu.qa/.../examination-rules-policy",
    "Library Usage": "https://www.udst.edu.qa/.../library-usage-policy",
    "Attendance Policy": "https://www.udst.edu.qa/.../attendance-policy",
    "Disciplinary Actions": "https://www.udst.edu.qa/.../disciplinary-actions-policy",
    "Data Privacy": "https://www.udst.edu.qa/.../data-privacy-policy",
    "Grading System": "https://www.udst.edu.qa/.../grading-system-policy",
    "Student Appeals": "https://www.udst.edu.qa/.../student-appeals-policy",
    "Harassment Policy": "https://www.udst.edu.qa/.../harassment-policy",
    "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Right to Refuse Service": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/right-refuse-service-procedure",
    "Library Study Room Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
    "Digital Media Centre Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/digital-media-centre-booking",
    "Use of Library Space": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy"
}

# fetch policy data
def fetch_policy_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("div")
    return tag.text.strip() if tag else ""

# chunking function
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# get embeddings
def get_text_embedding(text_chunks):
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=text_chunks)
    return [emb.embedding for emb in embeddings_batch_response.data]

# create FAISS index
def create_faiss_index(embeddings):
    embedding_vectors = np.array(embeddings, dtype=np.float32)
    embedding_vectors = normalize(embedding_vectors, axis=1)  # Normalize for better retrieval
    d = embedding_vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)  # HNSW for efficient search
    index.add(embedding_vectors)
    return index

# search FAISS index
def search_relevant_chunks(index, query_embedding, k=3):
    query_embedding = normalize(np.array([query_embedding], dtype=np.float32))
    D, I = index.search(query_embedding, k)
    return I[0]

# generate answer using Mistral
def mistral_answer(query, context):
    prompt = f"""
    You are an AI assistant trained to answer questions based on UDST policies.
    Below is the relevant policy information:
    ---------------------
    {context}
    ---------------------
    Please provide a concise, factual, and accurate response using only the information given.
    Query: {query}
    Answer:
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return response.choices[0].message.content

# streamlit UI
def streamlit_app():
    st.title('UDST Policy Chatbot')
    
    #user input query
    query = st.text_input("Enter your query:")
    
    if query:
        # intent classification
        labels = list(policies.keys())
        classification = classifier(query, labels)
        predicted_policy = classification['labels'][0]
        st.write(f"Predicted Policy: {predicted_policy}")
        
        # fetch and process policy data
        policy_text = fetch_policy_data(policies[predicted_policy])
        chunks = chunk_text(policy_text)
        embeddings = get_text_embedding(chunks)
        faiss_index = create_faiss_index(embeddings)
        
        # retrieve most relevant chunks
        query_embedding = get_text_embedding([query])[0]
        retrieved_chunks = [chunks[i] for i in search_relevant_chunks(faiss_index, query_embedding)]
        context = " ".join(retrieved_chunks)
        
        # generate answer
        answer = mistral_answer(query, context)
        st.text_area("Answer:", answer, height=200)

if __name__ == "__main__":
    streamlit_app()
