# knowledge_assistant.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx
# --- Configuration ---
OPENROUTER_API_KEY = "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"


LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 3  # Number of document chunks to retrieve
DOCUMENT_FILE_PATH = "faq.txt"

# --- Document Loading ---
def load_documents_from_file(filepath=DOCUMENT_FILE_PATH):
    """Loads documents from a text file, splitting by double newlines (paragraphs/blocks)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines, which typically separate paragraphs or Q&A blocks in the FAQ
        documents = content.split('\n\n')
        
        processed_docs = []
        for doc in documents:
            stripped_doc = doc.strip()
            # Keep if it's not empty and has a reasonable length (e.g., more than 5 words)
            if stripped_doc and len(stripped_doc.split()) > 5:
                # Skip standalone section headers (e.g., "## General Questions")
                if stripped_doc.startswith("## ") and "\n" not in stripped_doc:
                    continue
                processed_docs.append(stripped_doc)
        
        if not processed_docs:
            print(f"Warning: No valid document chunks loaded from {filepath}. The knowledge base will be empty.")
        else:
            print(f"Loaded {len(processed_docs)} document chunks from {filepath}.")
            # For debugging, uncomment to see the loaded chunks:
            # for i, chunk in enumerate(processed_docs):
            #     print(f"Chunk {i+1}:\n{chunk}\n------------------")
        return processed_docs
    except FileNotFoundError:
        print(f"Error: Document file '{filepath}' not found. Please create it.")
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

class DocumentStore:
    def __init__(self, documents, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()] # Ensure docs are non-empty strings
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        if self.documents:
            self._build_index()
        else:
            print("No valid documents provided to DocumentStore, skipping index build.")

    def _build_index(self):
        print(f"Building document index with {len(self.documents)} documents...")
        try:
            embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=True)
            
            if embeddings.ndim == 1: # Handle case of single document encoding
                 embeddings = embeddings.reshape(1, -1)
            
            if embeddings.shape[0] == 0:
                print("No embeddings generated. Index cannot be built.")
                self.index = None
                return

            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
            self.index.add(np.array(embeddings, dtype=np.float32))
            print(f"Index built successfully with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None


    def retrieve(self, query_text, top_k=TOP_K_RESULTS):
        if not self.index or self.index.ntotal == 0:
            print("Index not built or is empty. Cannot retrieve.")
            return []
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        if query_embedding.ndim == 1: # Ensure it's 2D for FAISS search
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        retrieved_docs = []
        for i in range(len(indices[0])):
            doc_index = indices[0][i]
            if 0 <= doc_index < len(self.documents):  # Check valid index
                retrieved_docs.append(self.documents[doc_index])
        return retrieved_docs

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
    def __init__(self, documents, openrouter_api_key):
        self.document_store = DocumentStore(documents)
        if not self.document_store.documents or not self.document_store.index:
            print("Warning: Document store is empty or index failed to build. Assistant may not function correctly.")
        self.llm_client = LLMClient(api_key=openrouter_api_key)

    def answer_question(self, question):
        print(f"\nUser Question: {question}")

        if not self.document_store.index or self.document_store.index.ntotal == 0:
             print("Knowledge base is not available.")
             return "I'm sorry, my knowledge base is currently unavailable. I cannot answer your question."

        # 1. Retrieve relevant information
        retrieved_context_list = self.document_store.retrieve(question)

        if not retrieved_context_list:
            print("No relevant context found in the knowledge base.")
            # Optionally, you could still try asking the LLM without context, or with a different prompt
            # For strict RAG, we return an "I don't know" style answer
            return "I couldn't find any relevant information in my knowledge base to answer that specific question."

        context_str = "\n\n---\n\n".join(retrieved_context_list) # Separate retrieved chunks clearly
        print(f"\n--- Retrieved Context ({len(retrieved_context_list)} chunk(s)) ---")
        print(context_str)
        print("------------------------------------")

        # 2. Prepare prompt for LLM
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

        # 3. Generate natural-language answer
        answer = self.llm_client.generate_answer(system_prompt, user_prompt_with_context)
        return answer

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Knowledge Assistant...")
    
    # Load documents from the FAQ file
    faq_documents = load_documents_from_file(DOCUMENT_FILE_PATH)

    if not faq_documents:
        print("Failed to load documents. The assistant might not work as expected. Please check 'company_faq.txt'.")
        # We can still proceed, but DocumentStore will be empty
    
    assistant = KnowledgeAssistant(documents=faq_documents, openrouter_api_key=OPENROUTER_API_KEY)

    # Example questions based on the FAQ
    print("\n--- Running Predefined Questions ---")
    example_questions = [
        "What does Innovatech Solutions specialize in?",
        "How can I contact support?",
        "Is there a free trial for InsightAI?",
        "Where are the headquarters of Innovatech Solutions located?",
        "What is the company culture like?",
        "How does your pricing work?",
        "Who is the CEO of Innovatech?" # This should ideally result in "I don't know"
    ]

    for q_idx, q in enumerate(example_questions):
        print(f"\n--- Question {q_idx+1}/{len(example_questions)} ---")
        response = assistant.answer_question(q)
        print(f"\nAssistant's Answer to '{q}':\n{response}\n{'='*60}")

    # Interactive mode
    print("\n--- Interactive Mode ---")
    print("Ask questions about Innovatech Solutions. Type 'exit' or 'quit' to end.")
    try:
        while True:
            user_query = input("\nYour question: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting Knowledge Assistant. Goodbye!")
                break
            if not user_query.strip():
                continue
            
            response = assistant.answer_question(user_query)
            print(f"\nAssistant's Answer:\n{response}\n{'='*60}")
    except KeyboardInterrupt:
        print("\nExiting due to user interruption. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred in interactive mode: {e}")