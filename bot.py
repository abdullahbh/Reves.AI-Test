# bot.py

import os
import time
import hashlib
import logging
import openai
import re
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAIError, RateLimitError

# LangChain and LangChain Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Subclass OpenAIEmbeddings to include batching and retry logic
class BatchedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        batch_size = 50  # Adjust as necessary
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
    Generate an MD5 hash for the given file.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_and_process_local_pdf(pdf_path):
    """
    Load the PDF, split into chunks, embed them, and create a FAISS vectorstore.
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

    vectorstore = FAISS.from_embeddings(list(zip(all_texts, all_embeddings)), embeddings, metadatas=metadatas)
    logging.info("Created FAISS vectorstore from embeddings.")
    return vectorstore

def create_rag_chain(vectorstore, llm):
    """
    Create a RetrievalQA chain using the given vectorstore and llm.
    Assumes system prompt and conversation flow logic are already in the chain prompt.
    """
    prompt_template = """You are a helpful company info assistant. 
You have the following instructions:
1. Present a brief company overview (max 2 sentences)
2. Then ask if the user wants to know more.
3. If yes, retrieve relevant info from the knowledge base (the provided context).
4. Offer to send the company profile via email.
5. If the user agrees, ask for their email address using the exact phrase: "Please provide your email address."
6. Use the context from the vector store for accurate info.

Context:
{context}

User Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    logging.info("Created RetrievalQA chain.")
    return rag_chain

def send_email(email_address):
    """
    Mock sending email. Replace this with actual email-sending logic.
    """
    logging.info(f"Sending company profile to {email_address}...")
    # Implement actual email sending logic here
    time.sleep(2)  # Simulate email sending delay
    logging.info("Email sent successfully.")

def is_valid_email(email):
    """
    Validate the email address using a simple regex.
    """
    regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(regex, email) is not None

class CompanyChatBot:
    def __init__(self, vectorstore):
        self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key())
        self.chain = create_rag_chain(vectorstore, self.llm)
        self.conversation_history = []
        self.waiting_for_email = False
        logging.info("Initialized CompanyChatBot.")

    def process_user_message(self, user_message):
        """
        Process the user's message and generate an appropriate response.
        """
        logging.info(f"Processing user message: {user_message}")

        # Handle explicit commands first
        explicit_commands = [
            "send me profile via email",
            "send profile to my email",
            "email me the profile",
            "send me the company profile",
            "send me profile via email",
            "can you send me the profile",
            "send me the profile via email",
            "please send me the profile via email",
            "could you send me the profile via email",
            "i would like to receive the profile via email",
            "send profile via email",
            "send me the profile by email"
        ]

        if user_message.lower() in explicit_commands:
            self.waiting_for_email = True
            response = "Sure! Could you please provide your email address?"
            logging.info("Triggering email prompt due to explicit command.")
            self.conversation_history.append(("user", user_message))
            self.conversation_history.append(("assistant", response))
            return response

        if self.waiting_for_email:
            email = user_message.strip()
            if is_valid_email(email):
                try:
                    send_email(email)
                    response = "Great! The company profile has been sent to your email address."
                    logging.info(f"Email sent to: {email}")
                except Exception as e:
                    response = "Sorry, there was an error sending the email. Please try again later."
                    logging.error(f"Error sending email to {email}: {e}")
                self.waiting_for_email = False
            else:
                response = "That doesn't look like a valid email. Please provide a valid email address."
                logging.warning(f"Invalid email provided: {email}")
            self.conversation_history.append(("user", user_message))
            self.conversation_history.append(("assistant", response))
            return response

        # Normal RAG response
        answer = self.chain.run(user_message)
        logging.info(f"Assistant response: {answer}")
        self.conversation_history.append(("user", user_message))
        self.conversation_history.append(("assistant", answer))

        # Enhanced trigger detection using regex
        email_prompt_patterns = [
            r"please provide your email",
            r"what is your email address",
            r"could you please provide your email",
            r"may i have your email",
            r"please share your email",
            r"provide your email",
            r"send you the profile via email",
            r"send the profile via email",
            r"send it via email",
            r"email me the profile",
            r"send the profile to my email"
        ]

        for pattern in email_prompt_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                self.waiting_for_email = True
                logging.info("Triggering email prompt based on assistant response.")
                break

        return answer
