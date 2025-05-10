import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import io # For handling uploaded file content
import httpx    
# --- Configuration (can be overridden by UI elements later) ---
# OPENROUTER_API_KEY will be fetched from st.secrets or user input
LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 3

# --- Document Loading ---
def load_documents_from_string(content_string):
    """Loads documents from a string, splitting by double newlines."""
    if not content_string:
        return []
    
    documents = content_string.split('\n\n')
    processed_docs = []
    for doc in documents:
        stripped_doc = doc.strip()
        if stripped_doc and len(stripped_doc.split()) > 5:
            if stripped_doc.startswith("## ") and "\n" not in stripped_doc:
                continue
            processed_docs.append(stripped_doc)
    
    if not processed_docs:
        st.warning("Warning: No valid document chunks found in the provided content. The knowledge base will be empty.")
    else:
        st.info(f"Loaded {len(processed_docs)} document chunks.")
    return processed_docs

# --- Caching for expensive resources ---
@st.cache_resource
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    st.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    st.info("Embedding model loaded.")
    return model

class DocumentStore:
    def __init__(self, documents, embedding_model_instance): # Takes model instance
        self.documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()]
        self.embedding_model = embedding_model_instance # Use passed instance
        self.index = None
        if self.documents:
            self._build_index()
        else:
            st.warning("No valid documents provided to DocumentStore, skipping index build.")

    def _build_index(self):
        st.info(f"Building document index with {len(self.documents)} documents...")
        try:
            # The model is already loaded, just encode
            embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=False)
            
            if embeddings.ndim == 1:
                 embeddings = embeddings.reshape(1, -1)
            
            if embeddings.shape[0] == 0:
                st.error("No embeddings generated. Index cannot be built.")
                self.index = None
                return

            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings, dtype=np.float32))
            st.success(f"Index built successfully with {self.index.ntotal} vectors.")
        except Exception as e:
            st.error(f"Error building FAISS index: {e}")
            self.index = None

    def retrieve(self, query_text, top_k=TOP_K_RESULTS):
        if not self.index or self.index.ntotal == 0:
            st.warning("Index not built or is empty. Cannot retrieve.")
            return []
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        retrieved_docs = []
        for i in range(len(indices[0])):
            doc_index = indices[0][i]
            if 0 <= doc_index < len(self.documents):
                retrieved_docs.append(self.documents[doc_index])
        return retrieved_docs

# Use st.cache_resource for DocumentStore if documents are stable (e.g. from a single upload session)
# The key for caching should ideally include the document content hash if it changes frequently.
# For simplicity, we'll re-create if documents change (new upload).
@st.cache_resource(show_spinner="Initializing Document Store...")
def get_document_store(_documents, _embedding_model): # _ to indicate these affect caching
    if not _documents:
        st.warning("Cannot initialize Document Store: No documents provided.")
        return None
    return DocumentStore(documents=_documents, embedding_model_instance=_embedding_model)


from openai import OpenAI

class LLMClient:
    def __init__(self, api_key, model_name=LLM_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_answer(self, system_prompt, user_prompt_with_context):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost",  # Optional
                "X-Title": "InnovatechKnowledgeAssistant",  # Optional
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_with_context},
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }

            response = httpx.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return "Sorry, I encountered an error trying to generate an answer."

class KnowledgeAssistant:
    def __init__(self, document_store_instance, llm_client_instance):
        self.document_store = document_store_instance
        self.llm_client = llm_client_instance
        if not self.document_store or not self.document_store.documents or (self.document_store.index is None or self.document_store.index.ntotal == 0) :
            st.warning("KnowledgeAssistant: Document store is empty or index failed to build. Assistant may not function correctly.")


    def answer_question(self, question, show_context=False):
        st.info(f"User Question: {question}")

        if not self.document_store or not self.document_store.index or self.document_store.index.ntotal == 0:
             st.error("Knowledge base is not available.")
             return "I'm sorry, my knowledge base is currently unavailable. I cannot answer your question."

        retrieved_context_list = self.document_store.retrieve(question)

        if not retrieved_context_list:
            st.warning("No relevant context found in the knowledge base.")
            return "I couldn't find any relevant information in my knowledge base to answer that specific question."

        context_str = "\n\n---\n\n".join(retrieved_context_list)
        if show_context:
            with st.expander(f"Retrieved Context ({len(retrieved_context_list)} chunk(s))", expanded=False):
                st.markdown(context_str)

        system_prompt = (
            "You are a helpful Knowledge Assistant for Innovatech Solutions. "
            "Answer the user's question based *only* on the provided context below. "
            "Be concise and directly answer the question. "
            "If the context does not contain the answer, state that you cannot answer based on the provided information. "
            "Do not make up information or use external knowledge. Quote relevant parts of the context if it helps clarity."
            "If the provided context *does not contain the specific information* to answer the question, you *must* clearly state that the information is not available in the provided documents. For example, respond with: 'The provided information does not specify [the detail requested].' or 'I cannot answer this question based on the context given.'"
        )
        
        user_prompt_with_context = f"""
Context from Innovatech Solutions' FAQ:
---
{context_str}
---

Question: {question}

Based *only* on the context provided above, what is the answer to the question?
Answer:"""

        with st.spinner("Generating answer from LLM..."):
            answer = self.llm_client.generate_answer(system_prompt, user_prompt_with_context)
        return answer

# --- Streamlit App UI ---
st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("📚 Knowledge Assistant for Innovatech Solutions")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # API Key
    # Try to get API key from st.secrets
    OPENROUTER_API_KEY = "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"

    openrouter_api_key = OPENROUTER_API_KEY

    

    # File Uploader
    uploaded_file = st.file_uploader("Upload your FAQ/Knowledge Base file (e.g., faq.txt)", type=['txt'])
    
    st.markdown("---")
    st.subheader("Options")
    show_retrieved_context = st.checkbox("Show retrieved context", value=False)
    
    st.markdown("---")
    st.markdown("Powered by [OpenRouter](https://openrouter.ai) and [Sentence Transformers](https://sbert.net).")


# --- Main App Logic ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'doc_store_initialized_for_file' not in st.session_state:
    st.session_state.doc_store_initialized_for_file = None


# Initialize embedding model once
embedding_model_instance = load_embedding_model()

# Process uploaded file and initialize assistant
if uploaded_file is not None:
    # Check if this is a new file or the same one
    if st.session_state.doc_store_initialized_for_file != uploaded_file.name:
        st.session_state.messages = [] # Reset chat if new file
        with st.spinner("Processing document..."):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_content = stringio.read()
            documents = load_documents_from_string(file_content)

            if documents and openrouter_api_key and embedding_model_instance:
                doc_store = get_document_store(tuple(documents), embedding_model_instance) # tuple for hashability
                if doc_store:
                    llm_client = LLMClient(api_key=openrouter_api_key)
                    st.session_state.assistant = KnowledgeAssistant(doc_store, llm_client)
                    st.session_state.doc_store_initialized_for_file = uploaded_file.name
                    st.success(f"Knowledge Assistant initialized with '{uploaded_file.name}'. Ready to answer questions!")
                else:
                    st.error("Failed to initialize Document Store. Assistant is not ready.")
                    st.session_state.assistant = None
            elif not documents:
                st.warning("No documents loaded from file. Assistant cannot be initialized.")
                st.session_state.assistant = None
            elif not openrouter_api_key:
                st.warning("OpenRouter API Key is missing. Assistant cannot be initialized.")
                st.session_state.assistant = None
            else: # Should not happen if embedding_model_instance is loaded
                 st.error("Embedding model not available. Assistant cannot be initialized.")
                 st.session_state.assistant = None
    # If it's the same file, and assistant is already initialized, do nothing here.
    # If API key changed, re-init LLM client part of assistant
    elif st.session_state.assistant and st.session_state.assistant.llm_client.api_key != openrouter_api_key:
        if openrouter_api_key:
            st.session_state.assistant.llm_client = LLMClient(api_key=openrouter_api_key)
            st.info("LLM Client API key updated.")
        else:
            st.warning("OpenRouter API Key removed. LLM client will not work.")


elif st.session_state.assistant is not None: # File removed after being loaded
    st.warning("FAQ file removed. Please upload a file to enable the assistant.")
    st.session_state.assistant = None
    st.session_state.doc_store_initialized_for_file = None
    st.session_state.messages = []


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
    elif st.session_state.assistant is None:
        st.warning("Please upload an FAQ/Knowledge Base file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = st.session_state.assistant.answer_question(prompt, show_context=show_retrieved_context)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Initial message if no file is uploaded
if not uploaded_file:
    st.info("Welcome! Please upload an FAQ or knowledge base .txt file and enter your OpenRouter API key to get started.")