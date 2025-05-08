# streamlit_agent_app.py

import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI as VanillaOpenAI
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain.chains import LLMMathChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Any, Dict, List, Union
from langchain_core.agents import AgentAction, AgentFinish

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model for the RAG system's generator LLM
RAG_LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
# Model for the LangChain Agent
AGENT_LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
# AGENT_LLM_MODEL_NAME = "mistralai/mistral-7b-instruct:free" # Faster alternative for agent

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 2
DOCUMENT_FILE_PATH = "company_faq.txt" # Make sure this file exists

# --- Streamlit Callback Handler ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.step_counter = 0

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.step_counter += 1
        self.container.info(f"**Step {self.step_counter}: Agent Decision**\n* Tool: `{action.tool}`\n* Input: `{action.tool_input}`\n* Thought: _{action.log.strip()}_")

    def on_tool_end(self, output: str, name: str, **kwargs: Any) -> Any:
        self.container.success(f"**Step {self.step_counter}: Tool `{name}` Result**\n```\n{output[:500]}...\n```")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        # The final answer will be displayed by the main app logic
        pass # No need to log final answer here, it's handled separately

# --- Caching for Expensive Operations ---
@st.cache_resource
def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

@st.cache_resource
def load_and_index_documents(filepath=DOCUMENT_FILE_PATH):
    print(f"Loading and indexing documents from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        documents = content.split('\n\n')
        processed_docs = []
        for doc in documents:
            stripped_doc = doc.strip()
            if stripped_doc and len(stripped_doc.split()) > 5:
                if stripped_doc.startswith("## ") and "\n" not in stripped_doc:
                    continue
                processed_docs.append(stripped_doc)
        
        if not processed_docs:
            st.error(f"No valid document chunks loaded from {filepath}. RAG tool will be ineffective.")
            return None, None

        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode(processed_docs, convert_to_tensor=False, show_progress_bar=False)
        
        if embeddings.ndim == 1: embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] == 0: 
            st.error("No embeddings generated. Index cannot be built.")
            return processed_docs, None

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        print(f"Document index built with {index.ntotal} vectors.")
        return processed_docs, index
    except FileNotFoundError:
        st.error(f"Error: Document file '{filepath}' not found. Please create it.")
        return None, None
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        return None, None

# --- RAG Components (modified for caching and Streamlit context) ---
class DocumentStore:
    def __init__(self):
        # Documents and index are loaded once via st.cache_resource
        self.documents, self.index = load_and_index_documents()
        self.embedding_model = get_embedding_model()
        if self.documents is None or self.index is None:
            st.warning("Document store could not be initialized properly.")

    def retrieve(self, query_text, top_k=TOP_K_RESULTS):
        if not self.index or self.index.ntotal == 0 or not self.documents:
            return []
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.documents[idx] for idx in indices[0] if 0 <= idx < len(self.documents)]

class RagLLMClient:
    def __init__(self, api_key, model_name=RAG_LLM_MODEL_NAME):
        self.client = VanillaOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name

    def generate_answer(self, system_prompt, user_prompt_with_context):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_with_context},
                ],
                temperature=0.2, max_tokens=350 # Slightly more tokens for web display
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling OpenRouter API for RAG: {e}")
            return "Sorry, RAG system encountered an error generating a response."

class KnowledgeAssistant:
    def __init__(self, openrouter_api_key):
        self.document_store = DocumentStore() # Uses cached documents/index
        self.llm_client = RagLLMClient(api_key=openrouter_api_key)

    def query_knowledge_base(self, question: str) -> str:
        if not self.document_store.index or self.document_store.index.ntotal == 0:
             return "Knowledge base is not available or empty. Please check the document file."

        retrieved_context_list = self.document_store.retrieve(question)
        if not retrieved_context_list:
            return "I couldn't find relevant information in the Innovatech knowledge base for that specific question."

        context_str = "\n\n---\n\n".join(retrieved_context_list)
        system_prompt = (
            "You are a helpful Knowledge Assistant for Innovatech Solutions. "
            "Answer the user's question based *only* on the provided context below. "
            "Be concise. If the context does not contain the answer, state that clearly (e.g., 'The provided information does not specify...'). "
            "Do not make up information or use external knowledge."
        )
        user_prompt_with_context = f"Context from Innovatech Solutions' FAQ:\n---\n{context_str}\n---\n\nQuestion: {question}\n\nAnswer:"
        
        raw_llm_answer = self.llm_client.generate_answer(system_prompt, user_prompt_with_context)
        if not raw_llm_answer or len(raw_llm_answer.strip()) < 5:
            return f"The provided information from Innovatech Solutions' FAQ does not seem to contain an answer to: '{question}'."
        return raw_llm_answer

# --- LangChain Agent Setup (cached) ---
@st.cache_resource
def get_agent_executor(_openrouter_api_key, _streamlit_callback_container): # Underscore to show they are for initialization
    if not _openrouter_api_key:
        st.error("OpenRouter API Key is not configured. Agent cannot be initialized.")
        return None

    agent_llm = ChatOpenAI(
        model=AGENT_LLM_MODEL_NAME,
        openai_api_key=_openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        streaming=False # Keep false for agent thoughts, streaming can be for final answer if desired
    )

    knowledge_assistant_instance = KnowledgeAssistant(openrouter_api_key=_openrouter_api_key)

    calculator_chain = LLMMathChain.from_llm(llm=agent_llm, verbose=False)
    calculator_tool = Tool(
        name="calculator", func=calculator_chain.run,
        description="Useful for math calculations or quantitative problems. Input: math expression."
    )

    def define_word_function(word_or_phrase: str) -> str:
        try:
            response = agent_llm.invoke([
                SystemMessage(content="You are a helpful dictionary. Provide a concise definition."),
                HumanMessage(content=f"Define: {word_or_phrase}")
            ])
            return response.content
        except Exception as e: return f"Dictionary tool error: {e}"
    dictionary_tool = Tool(
        name="dictionary_definer", func=define_word_function,
        description="Useful for defining words or phrases. Input: word/phrase to define."
    )

    rag_tool = Tool(
        name="innovatech_knowledge_base",
        func=knowledge_assistant_instance.query_knowledge_base,
        description="Use for questions about Innovatech Solutions (products, services, support, careers, etc.)."
    )
    tools = [calculator_tool, dictionary_tool, rag_tool]

    agent_prompt_template_str = """
    You are a helpful assistant. Tools: {tools}. Format:
    Thought: Do I need a tool? Yes
    Action: One of [{tool_names}]
    Action Input: Input to action
    Observation: Result
    ... (repeat Thought/Action/Action Input/Observation N times)
    Thought: I have the answer.
    Final Answer: The final answer.

    Hints:
    - "calculate", "how much is": use 'calculator'.
    - "define", "what does X mean": use 'dictionary_definer'.
    - Questions about 'Innovatech Solutions': use 'innovatech_knowledge_base'.
    - If unsure, try 'innovatech_knowledge_base' or answer directly if trivial.

    Begin! Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_prompt_template_str),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_structured_chat_agent(llm=agent_llm, tools=tools, prompt=prompt)
    
    # Pass the Streamlit container to the callback handler
    streamlit_callback = StreamlitCallbackHandler(container=_streamlit_callback_container)
    
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, callbacks=[streamlit_callback],
        handle_parsing_errors="True_then_retry_with_partial_output_if_error_persists_then_finish" 
        # More robust error handling
    )
    print("Agent Executor initialized.")
    return agent_executor

# --- Streamlit App UI ---
st.set_page_config(page_title="Innovatech Knowledge Assistant", layout="wide")
st.title("🤖 Innovatech Solutions Knowledge Assistant")
st.caption("Powered by RAG, LangChain Agents, and OpenRouter LLMs")

# Sidebar for API Key (optional, better to use .env)
# with st.sidebar:
#     st.header("Configuration")
#     # Manual API key input if .env is not preferred for some reason
#     # openrouter_api_key_input = st.text_input("OpenRouter API Key", type="password", value=OPENROUTER_API_KEY or "")
#     # if openrouter_api_key_input:
#     #     OPENROUTER_API_KEY = openrouter_api_key_input # Override if provided
    
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_thoughts" not in st.session_state:
    st.session_state.agent_thoughts = [] # To store agent actions for display

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a container for agent thoughts/actions
thoughts_container = st.expander("Agent's Thought Process (Live Updates)", expanded=False)

# Get or initialize agent_executor
# The _streamlit_callback_container is passed here for the callback handler
agent_executor = get_agent_executor(OPENROUTER_API_KEY, thoughts_container)

if agent_executor is None:
    st.error("Agent could not be initialized. Please check configurations and API key.")
    st.stop()


# React to user input
if prompt := st.chat_input("Ask about Innovatech, calculate, or define something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Clear previous thoughts for the new query
        thoughts_container.empty() # Clear previous content in expander
        st.session_state.agent_thoughts = [] # Reset internal thoughts log

        with st.spinner("Thinking and processing..."):
            try:
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": [
                        AIMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"])
                        for msg in st.session_state.messages[:-1] # Pass previous messages for context
                    ]
                })
                assistant_response = response.get("output", "Sorry, I could not generate a response.")
            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                assistant_response = "An error occurred while processing your request."
        
        st.markdown(assistant_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# --- Add a section to show the company_faq.txt content for reference ---
with st.sidebar:
    st.header("Knowledge Base Content")
    try:
        with open(DOCUMENT_FILE_PATH, 'r', encoding='utf-8') as f:
            faq_content = f.read()
        st.text_area("FAQ Document (`company_faq.txt`)", faq_content, height=300, disabled=True)
    except FileNotFoundError:
        st.warning(f"`{DOCUMENT_FILE_PATH}` not found. RAG will not have data.")
    except Exception as e:
        st.error(f"Could not read FAQ file: {e}")