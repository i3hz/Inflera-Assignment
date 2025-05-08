
# knowledge_assistant_with_agent.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI as VanillaOpenAI # For our RAG LLM
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI # For the agent's LLM
from langchain.agents import AgentExecutor, create_structured_chat_agent # Or create_openai_functions_agent
from langchain_core.tools import Tool
from langchain.chains import LLMMathChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any, Dict, List, Union
from langchain_core.agents import AgentAction, AgentFinish

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please create a .env file.")

# Model for the RAG system's generator LLM
RAG_LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
# Model for the LangChain Agent (can be the same or different)
AGENT_LLM_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free" # Or a smaller, faster one for agent decisions
# AGENT_LLM_MODEL_NAME = "mistralai/mistral-7b-instruct:free" # Example of a faster agent model

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 2
DOCUMENT_FILE_PATH = "company_faq.txt"

# --- Callback Handler for Logging Agent Decisions ---
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"\n تصمیم عامل [Decision Log]: Agent decided to use tool '{action.tool}' with input '{action.tool_input}'")
        print(f"  Thought Process: {action.log.strip()}")

    def on_tool_end(self, output: str, name: str, **kwargs: Any) -> Any:
        print(f" عامل انجام شد [Decision Log]: Tool '{name}' finished, output: '{output[:200]}...'")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        print(f"\n پاسخ نهایی عامل [Decision Log]: Agent finished. Final Answer: {finish.return_values['output']}")


# --- Document Loading (same as before) ---
def load_documents_from_file(filepath=DOCUMENT_FILE_PATH):
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
        if not processed_docs: print(f"Warning: No valid document chunks loaded from {filepath}.")
        else: print(f"Loaded {len(processed_docs)} document chunks from {filepath}.")
        return processed_docs
    except FileNotFoundError: print(f"Error: Document file '{filepath}' not found."); return []
    except Exception as e: print(f"Error loading documents: {e}"); return []

# --- DocumentStore (same as before) ---
class DocumentStore:
    def __init__(self, documents, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()]
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        if self.documents: self._build_index()
        else: print("No valid documents provided to DocumentStore, skipping index build.")

    def _build_index(self):
        print(f"Building document index with {len(self.documents)} documents...")
        try:
            embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=False)
            if embeddings.ndim == 1: embeddings = embeddings.reshape(1, -1)
            if embeddings.shape[0] == 0: self.index = None; return
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings, dtype=np.float32))
            print(f"Index built successfully with {self.index.ntotal} vectors.")
        except Exception as e: print(f"Error building FAISS index: {e}"); self.index = None

    def retrieve(self, query_text, top_k=TOP_K_RESULTS):
        if not self.index or self.index.ntotal == 0: return []
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.documents[idx] for idx in indices[0] if 0 <= idx < len(self.documents)]

# --- RAG LLM Client (using VanillaOpenAI for direct call) ---
class RagLLMClient:
    def __init__(self, api_key, model_name=RAG_LLM_MODEL_NAME):
        self.client = VanillaOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name

    def generate_answer(self, system_prompt, user_prompt_with_context):
        # print(f"\n--- RAG LLM ({self.model_name}) ---") # Less verbose now
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_with_context},
                ],
                temperature=0.2, max_tokens=300
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter API for RAG: {e}")
            return "Sorry, RAG system encountered an error."

# --- Knowledge Assistant (modified to be callable by a tool) ---
class KnowledgeAssistant:
    def __init__(self, documents, openrouter_api_key):
        self.document_store = DocumentStore(documents)
        self.llm_client = RagLLMClient(api_key=openrouter_api_key)
        if not self.document_store.documents or not self.document_store.index:
            print("Warning: RAG Document store empty or index failed.")

    def query_knowledge_base(self, question: str) -> str: # Renamed for clarity as a tool function
        # print(f"\nKnowledgeBase Tool activated for: {question}") # Logged by callback now
        if not self.document_store.index or self.document_store.index.ntotal == 0:
             return "Knowledge base is not available."

        retrieved_context_list = self.document_store.retrieve(question)
        if not retrieved_context_list:
            return "I couldn't find relevant information in the knowledge base for that question."

        context_str = "\n\n---\n\n".join(retrieved_context_list)
        system_prompt = (
            "You are a helpful Knowledge Assistant for Innovatech Solutions. "
            "Answer the user's question based *only* on the provided context below. "
            "Be concise. If the context does not contain the answer, state that. "
            "Do not make up information or use external knowledge."
            "If the provided context *does not contain the specific information* to answer the question, you *must* clearly state that the information is not available in the provided documents. For example, respond with: 'The provided information does not specify [the detail requested].' or 'I cannot answer this question based on the context given.'"
        )
        user_prompt_with_context = f"Context:\n---\n{context_str}\n---\n\nQuestion: {question}\n\nAnswer:"
        return self.llm_client.generate_answer(system_prompt, user_prompt_with_context)

# --- Initialize LangChain Agent Components ---
# Agent's LLM (can be different from RAG LLM)
agent_llm = ChatOpenAI(
    model=AGENT_LLM_MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1, # Agent needs to be precise
    streaming=False # Simpler for this example, can be True for interactive
)

# Initialize RAG system (needed for the RAG tool)
faq_documents = load_documents_from_file(DOCUMENT_FILE_PATH)
knowledge_assistant = KnowledgeAssistant(documents=faq_documents, openrouter_api_key=OPENROUTER_API_KEY)

# Define Tools
# 1. Calculator Tool
calculator_chain = LLMMathChain.from_llm(llm=agent_llm, verbose=False) # verbose=False to reduce LLMMathChain's own prints
calculator_tool = Tool(
    name="calculator",
    func=calculator_chain.run,
    description="""Useful for when you need to answer questions about math, perform calculations, 
    or solve quantitative problems. Input should be a clear mathematical question or expression.
    Examples: 'what is 5 plus 7?', 'calculate 12 * 4', 'square root of 64'."""
)

# 2. Dictionary Tool (using LLM to define)
def define_word_function(word_or_phrase: str) -> str:
    # print(f"\nDictionary Tool activated for: {word_or_phrase}") # Logged by callback
    try:
        # Use the agent's LLM to define the word.
        # A more robust solution might use a dedicated dictionary API.
        response = agent_llm.invoke([
            SystemMessage(content="You are a helpful dictionary. Provide a concise definition for the given word or phrase. If it's an idiom or common phrase, explain its meaning."),
            HumanMessage(content=f"Define: {word_or_phrase}")
        ])
        return response.content
    except Exception as e:
        print(f"Error in dictionary tool: {e}")
        return "Sorry, I couldn't define that word at the moment."

dictionary_tool = Tool(
    name="dictionary_definer",
    func=define_word_function,
    description="""Useful for when you need to define a word, get the meaning of a term, 
    or understand an idiom. Input should be the word or phrase to define.
    Examples: 'define serendipity', 'what does ubiquitous mean?', 'meaning of 'beat around the bush''"""
)

# 3. RAG Tool
rag_tool = Tool(
    name="innovatech_knowledge_base",
    func=knowledge_assistant.query_knowledge_base, # Use the method from our existing KA
    description="""Useful for answering questions about Innovatech Solutions, its products (like InsightAI, CloudFlow), 
    services, policies, company information, customer support, careers, or billing. 
    Use this for any question that is not a calculation or a request for a word definition.
    Examples: 'What does Innovatech Solutions do?', 'How to contact support?', 'tell me about InsightAI'"""
)

tools = [calculator_tool, dictionary_tool, rag_tool]

# Agent Prompt
# Using `create_structured_chat_agent` which works well with many chat models.
# It expects a prompt with "input" and "agent_scratchpad" variables.
# The agent_scratchpad is where the agent's intermediate thoughts and tool calls/responses go.
AGENT_PROMPT_TEMPLATE = """
You are a helpful assistant that can use tools to answer questions.
You have access to the following tools:

{tools}

Use the following format for your thought process and final answer:

Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to answer the user's question.
Final Answer: The final answer to the original input question.

If the query contains keywords suggesting calculation (e.g., "calculate", "how much is", "what is X plus Y"), consider using the 'calculator' tool.
If the query asks for a definition or meaning of a word/phrase (e.g., "define", "what does X mean"), consider using the 'dictionary_definer' tool.
For all other questions, especially those related to 'Innovatech Solutions', its products, or general company information, use the 'innovatech_knowledge_base' tool.
If you are unsure or none of the tools seem appropriate, try to answer directly or state you cannot answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history", optional=True), # For conversational memory, if added
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the Agent
# Using create_structured_chat_agent
agent = create_structured_chat_agent(
    llm=agent_llm,
    tools=tools,
    prompt=prompt
)

# Create Agent Executor
logging_callback = LoggingCallbackHandler()
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False, # We have our custom logger, LangChain's verbose can be noisy
    callbacks=[logging_callback],
    handle_parsing_errors=True # Important for robustness
)


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- LangChain Agent Interactive Mode ---")
    print("Keywords: 'calculate', 'define'. Otherwise RAG. Type 'exit' or 'quit' to end.")

    # Pre-load RAG documents (if not already done by global scope for KnowledgeAssistant)
    if not faq_documents and DOCUMENT_FILE_PATH:
        print("Attempting to load FAQ documents for RAG tool...")
        knowledge_assistant.document_store.documents = load_documents_from_file(DOCUMENT_FILE_PATH)
        if knowledge_assistant.document_store.documents:
            knowledge_assistant.document_store._build_index()
        else:
            print("CRITICAL: FAQ documents still not loaded for RAG. RAG tool may not work.")


    example_agent_questions = [
        "What does Innovatech Solutions specialize in?",
        "Calculate 25 * (4 + 6)",
        "Define the word 'ephemeral'",
        "How can I contact Innovatech customer support?",
        "What is the square root of 144 plus 10?",
        "What's the meaning of 'throw in the towel'?",
        "Tell me about the free trial for InsightAI.",
        "Who is the president of the USA?" # Should use RAG and likely say "I don't know based on context"
    ]

    for q_idx, q_text in enumerate(example_agent_questions):
        print(f"\n--- Agent Question {q_idx+1}/{len(example_agent_questions)} ---")
        print(f"User Query: {q_text}")
        try:
            response = agent_executor.invoke({"input": q_text})
            # The final answer is already printed by the on_agent_finish callback
        except Exception as e:
            print(f"Error during agent execution for '{q_text}': {e}")
        print(f"\n{'='*70}")


    print("\n--- Interactive Agent Session ---")
    try:
        while True:
            user_query = input("\nYour question for the Agent: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting Agent. Goodbye!")
                break
            if not user_query.strip():
                continue
            
            try:
                response = agent_executor.invoke({"input": user_query})
                # Final answer printed by callback
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"\n{'='*70}")

    except KeyboardInterrupt:
        print("\nExiting due to user interruption. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred in interactive agent mode: {e}")