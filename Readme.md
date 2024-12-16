# Reves.AI-Test

Reves.AI-Test is an intelligent chatbot agent that uses Retrieval-Augmented Generation (RAG) to provide company information and handle email communications. It integrates LangChain for RAG implementation, WebSockets for real-time communication, and a Streamlit-based user interface.

---

## Features

### Core Features

1. **Conversational Flow**
   * Presents a brief company overview.
   * Retrieves detailed information from a knowledge base on user request.
   * Offers to send the company profile via email and validates email addresses.
   * Maintains a natural and logical conversation flow.
2. **RAG Implementation**
   * Processes PDF documents to create vector embeddings.
   * Uses **FAISS** for vector search and retrieval.
   * Generates contextual responses based on retrieved information.
3. **Real-time Chat Interface**
   * **WebSocket**-based backend for live communication.
   * Streamlit front-end for a simple and interactive user interface.
4. **Email Functionality**
   * Simulates sending company profiles to user-provided email addresses.
   * Logs email details for debugging and verification.

---

## Technical Stack

### Backend

* **Python**
* **LangChain** for RAG implementation
* **FAISS** for vector storage and retrieval
* **WebSocket** for real-time communication

### Frontend

* **Streamlit** for user interface

### Other Tools

* **OpenAI** API for embeddings and conversational LLMs
* **dotenv** for environment variable management

---

## Installation and Setup

### Prerequisites

* Python 3.8+
* OpenAI API Key
* Required libraries (see `requirements.txt`)

### Steps

1. Clone the repository:

   ```
   git clone https://github.com/abdullahbh/Reves.AI-Test.git
   cd Reves.AI-Test
   ```
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Add your OpenAI API Key to a `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Start the WebSocket server:

   ```
   python websocket.py
   ```
5. Launch the Streamlit frontend:

   ```
   streamlit run frontend.py
   ```
6. Open your browser and navigate to the Streamlit app URL (usually `http://localhost:8501`).

---

## Usage

1. Enter your question in the chat box.
2. The bot will respond with relevant company information.
3. Request the company profile via email by providing your email address.
4. The email details will be logged for debugging purposes.

---

## Architecture Overview

1. **WebSocket Server (**`websocket.py`**)**
   * Handles real-time communication between the frontend and backend.
   * Manages chatbot initialization and user message processing.
2. **Streamlit Frontend (**`frontend.py`**)**
   * Provides a user-friendly interface for interaction.
   * Sends user queries to the WebSocket server and displays bot responses.
3. **RAG Implementation (**`bot.py`**)**
   * Loads and processes the company PDF document.
   * Splits the document into chunks, generates embeddings, and stores them in FAISS.
   * Uses a LangChain RetrievalQA chain to answer user queries.
4. **Email Handling (**`bot.py`**)**
   * Validates and logs email-sending requests.

---

## Known Limitations

* Email functionality is a mock implementation; real email-sending logic is not included.
* PDF processing assumes a single file with a specific format.
* The WebSocket server runs locally and does not support authentication or encryption.
* Streamlit frontend is minimal and may require enhancement for production use.

---

## Future Improvements

* Add real email-sending functionality (e.g., using `smtplib` or third-party APIs like SendGrid).
* Support multiple PDF files and dynamic vectorstore updates.
* Improve security with WebSocket authentication and HTTPS.
* Enhance the frontend UI/UX with advanced features and styling.
* Optimize embeddings for larger datasets with better batching strategies.

---

## Demo

A video demonstrating the functionality is available [here](https://drive.google.com/file/d/13NUXYMuLVbRUInqSNdhxpkgMAP7y9XN7/view?usp=sharing).

## RAG Implementation (Brief)

1. **PDF Processing with Hashing**
   * Generates an MD5 hash of the PDF to detect changes.
   * Regenerates embeddings only if the PDF is updated, saving computation time.
2. **Embedding and Storage**
   * Splits PDF into chunks (500 characters with 100 overlap).
   * Embeds chunks using OpenAI's embedding API.
   * Stores embeddings and metadata in a FAISS vectorstore.
3. **RetrievalQA Chain**
   * Retrieves relevant text chunks from FAISS.
   * Uses GPT-based LLM to generate responses based on retrieved context and user queries.
4. **Efficiency**
   * Local storage and hashing ensure reusability and reduced redundancy.


Feel free to reach out for any issues or further clarifications!
