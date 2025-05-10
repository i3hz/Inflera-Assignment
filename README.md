

# Innovatech AI Knowledge Assistant

This project implements an AI-powered knowledge assistant for "Innovatech Solutions," a fictional company. The assistant can:

1.  **Answer questions based on a local document collection (FAQ file) using RAG (Retrieval Augmented Generation).**
2.  **Perform calculations using an LLM-powered calculator tool.**
3.  **Define words and phrases using an LLM-powered dictionary tool.**
4.  **Route user queries to the appropriate tool using a LangChain agent.**
5.  **Provide a user-friendly web interface via Streamlit.**

The system leverages OpenRouter to access various Large Language Models (LLMs) for generation and agentic reasoning.

## Features

*   **RAG Pipeline:**
    *   Loads company information from `company_faq.txt`.
    *   Uses `sentence-transformers` for embedding documents and queries.
    *   Utilizes `FAISS` for efficient similarity search in a vector store.
    *   Generates answers using an LLM, grounded in the retrieved context.
*   **Agentic Tool Usage (LangChain):**
    *   **Calculator Tool:** For mathematical queries.
    *   **Dictionary Tool:** For defining terms.
    *   **Knowledge Base Tool:** For company-specific questions (interfaces with the RAG pipeline).
    *   An LLM-based agent decides which tool to use based on the user's query.
*   **Streamlit Web Interface:**
    *   Interactive chat UI for querying the assistant.
    *   Real-time display of the agent's thought process and tool usage.
    *   Displays the content of the knowledge base (`company_faq.txt`) for reference.
    *   Caches models and data for improved performance.
*   **OpenRouter Integration:** Configured to use LLMs via the OpenRouter API.

## Project Structure
.
├── streamlit_agent_app.py # Main Streamlit application script
├── company_faq.txt # Knowledge base document for Innovatech Solutions
├── .env # For API keys and environment variables (Gitignored)
└── README.md 

## Prerequisites

*   Python 3.8+
*   An OpenRouter API Key

## Setup

1.  **Clone the repository (or create the files):**
    If this were a Git repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
    Otherwise, ensure `streamlit_agent_app.py` and `company_faq.txt` are in the same directory.

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit openai python-dotenv sentence-transformers faiss-cpu numpy langchain langchain-openai langchain-community
    ```

4.  **Set up your OpenRouter API Key:**
    Create a file named `.env` in the project's root directory and add your OpenRouter API key:
    ```env
    # .env
    OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
    ```

5.  **Prepare the Knowledge Base:**
    The `company_faq.txt` file contains the information the assistant will use for company-specific questions. You can customize this file with your own data. Ensure it includes information you want the assistant to know (e.g., CEO name, product details). A sample `company_faq.txt` structure is:

    ```text
    # company_faq.txt

    ## General Questions

    **Q: What does Innovatech Solutions do?**
    A: Innovatech Solutions specializes in developing cutting-edge software for businesses...

    **Q: Where is Innovatech Solutions located?**
    A: Our headquarters are located at 123 Tech Park Drive, Silicon Valley, CA 94000...

    ## Leadership

    **Q: Who is the CEO of Innovatech Solutions?**
    A: The CEO of Innovatech Solutions is Dr. Aris Thorne...

    (Add more Q&A pairs or information as needed)
    ```

## Running the Application

Once the setup is complete, run the Streamlit application from your terminal:

```bash
streamlit run streamlit_agent_app.py
This will start the Streamlit server, and the application should open automatically in your default web browser. If not, the terminal will provide a URL (usually http://localhost:8501).
How to Use

    Open the web application in your browser.

    Type your question into the chat input at the bottom of the page.

    The assistant will process your query:

        The agent's "thought process" (tool selection, inputs, observations) will be displayed in an expandable section.

        The final answer will appear in the chat.

    Try different types of queries:

        Company Info: "What services does Innovatech offer?" or "Who is the CEO?"

        Calculations: "Calculate 15 * (3 + 7)"

        Definitions: "Define 'synergy'"

Configuration

    LLM Models: The LLMs used for the RAG generator and the agent can be configured at the top of streamlit_agent_app.py:

        RAG_LLM_MODEL_NAME

        AGENT_LLM_MODEL_NAME
        (Ensure these models are available via your OpenRouter plan.)

    Embedding Model: EMBEDDING_MODEL_NAME can also be changed if desired.

    Knowledge Base File: The DOCUMENT_FILE_PATH variable points to company_faq.txt.

Potential Improvements & Future Work

    More Sophisticated Document Chunking: For larger or more complex documents, implement more advanced text splitting strategies.

    Advanced Vector Store: For production, consider using a persistent vector database like ChromaDB, Pinecone, or Weaviate.

    Conversational Memory: Enhance the agent's prompt and chat history management for more robust multi-turn conversations.

    Streaming Responses: Implement streaming for the final LLM answer for a more responsive UI.

    Error Handling: Add more comprehensive error handling and user feedback.

    Evaluation: Implement an evaluation framework to measure the RAG system's and agent's performance.

    More Tools: Expand the agent's capabilities by adding more tools (e.g., web search, database query).

    User Authentication: For private knowledge bases.

Troubleshooting

    OPENROUTER_API_KEY not found: Ensure your .env file is correctly named, in the root directory, and contains the valid API key.

    company_faq.txt not found: Make sure the file exists in the same directory as streamlit_agent_app.py or update DOCUMENT_FILE_PATH.

    Slow performance on first run: Model loading and index building can take time initially. Subsequent interactions should be faster due to caching.

    Agent not picking the right tool:

        Review and refine the description for each Tool.

        Adjust the agent's system prompt (in streamlit_agent_app.py) to provide clearer instructions or hints.

    RAG tool says "I don't know":

        Ensure the information is actually present in company_faq.txt.

        The RAG system is designed to not hallucinate if the info isn't in its context.
