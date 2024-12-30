import os
import time
import hashlib
import logging
import openai
import re
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAIError, RateLimitError

# LangChain / Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def api_key():
    """
    Load the OpenAI API key from environment variables.
    """
    load_dotenv(override=True)
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        logging.error("OPENAI_API_KEY not found in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY not found.")
    openai.api_key = key
    return key


class BatchedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Subclass of OpenAIEmbeddings with batch processing & retry logic.
    """
    def embed_documents(self, texts):
        batch_size = 50
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Documents"):
            batch = texts[i:i+batch_size]
            retries = 0
            max_retries = 5
            while retries < max_retries:
                try:
                    batch_embeddings = super().embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except RateLimitError:
                    wait_time = 2 ** retries
                    logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                except OpenAIError as e:
                    logging.error(f"OpenAI API error: {e}")
                    raise e
            else:
                raise Exception("Max retries exceeded for embedding requests.")
        return embeddings


def file_hash(file_path):
    """
    Generate an MD5 hash for the given file to check for updates.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_and_process_local_pdf(pdf_path):
    """
    Load a PDF, split into chunks, embed them, and create a FAISS vector store.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} documents from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    logging.info(f"Split documents into {len(splits)} chunks.")

    embeddings = BatchedOpenAIEmbeddings(openai_api_key=api_key())

    all_texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    all_embeddings = embeddings.embed_documents(all_texts)
    logging.info("Generated embeddings for all document chunks.")

    vectorstore = FAISS.from_embeddings(
        list(zip(all_texts, all_embeddings)),
        embeddings,
        metadatas=metadatas
    )
    logging.info("Created FAISS vector store from embeddings.")
    return vectorstore


def create_retriever_tool(vectorstore):
    """
    Build a RetrievalQA chain and wrap it as a Tool for the Agent.
    """
    prompt_template = """You are a helpful assistant with access to a company knowledge base.
Use the context from the knowledge base to accurately answer questions about the company.

Context:
{context}

Question:
{question}

Answer concisely:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key(), temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    def _search_company_info(query: str) -> str:
        """
        This function is called when the Agent decides to 
        query the knowledge base about company info.
        """
        return qa_chain.run(query)

    return Tool(
        name="Search Company Knowledge Base",
        func=_search_company_info,
        description=(
            "Use this tool to answer questions about the company, "
            "retrieve data from the PDF knowledge base, etc. "
            "Input: any question about the company."
        ),
    )


def send_email_tool(email_address: str) -> str:
    """
    Mock function to simulate sending email with company profile.
    Replace with your real email-sending logic if needed.
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(pattern, email_address.strip()):
        return "That doesn't look like a valid email format. Please provide a valid email address."

    logging.info(f"Sending the company profile to {email_address} (mock).")
    time.sleep(2)  # simulating a delay
    return f"The company profile has been sent to {email_address}."


def initialize_agentic_bot(pdf_path="company_profile.pdf", store_path="my_vectorstore"):
    """
    Load/update the FAISS store, build Tools, and create a conversational Agent 
    that can either provide info or send an email, while keeping the conversation going.
    """
    # ---- Step 1: Load / check VectorStore ----
    hash_path = os.path.join(store_path, "hash.txt")
    pdf_hash = file_hash(pdf_path)
    saved_hash = None

    if os.path.exists(store_path) and os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            saved_hash = f.read().strip()

    if pdf_hash != saved_hash or not os.path.exists(store_path):
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        vectorstore = load_and_process_local_pdf(pdf_path)
        vectorstore.save_local(store_path)
        with open(hash_path, "w") as f:
            f.write(pdf_hash)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key())
        vectorstore = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ---- Step 2: Create Tools ----
    search_tool = create_retriever_tool(vectorstore)
    email_tool = Tool(
        name="Send Email",
        func=send_email_tool,
        description=(
            "Use this tool to send the company profile via email. "
            "Input: a valid email address as a string."
        ),
    )

    # ---- Step 3: Customize Agent's system message to keep conversation going ----
    # This 'prefix' or 'system_message' instructs the agent to be proactive:
    system_prefix = """You are an AI assistant that can do two main tasks:
1) Provide or retrieve information about the company from a knowledge base.
2) Send the company profile via email.

You must keep the conversation going if the user does not explicitly end it. 
- If the user might want more info, offer it.
- If the user mentions or requests an email, confirm or ask for the address, then call the "Send Email" tool.
- If the user has a question about the company, call "Search Company Knowledge Base" to find an answer.
- Stay friendly and helpful. 
"""

    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key(), temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=[search_tool, email_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        agent_kwargs={
            "system_message": system_prefix
        },
    )

    return agent


# Optional: A wrapper class for convenience, similar to your original approach
class CompanyAgentBot:
    """
    A wrapper around the agent for easier usage from websocket.py.
    """
    def __init__(self, agent):
        self.agent = agent

    def process_message(self, user_input: str) -> str:
        try:
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            logging.error(f"Error in agent run: {e}")
            return f"Sorry, an error occurred: {e}"
