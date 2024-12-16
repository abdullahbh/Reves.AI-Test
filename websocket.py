import asyncio
import websockets
import json
from bot import CompanyChatBot, load_and_process_local_pdf, file_hash, api_key
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

# Initialize the chatbot with the vectorstore
async def initialize_chatbot(pdf_path="company_profile.pdf", store_path="my_vectorstore"):
    hash_path = os.path.join(store_path, "hash.txt")
    pdf_hash = file_hash(pdf_path)

    # Check if vectorstore exists
    if os.path.exists(store_path) and os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            saved_hash = f.read().strip()
    else:
        saved_hash = None

    if pdf_hash != saved_hash or not os.path.exists(store_path):
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        vectorstore = load_and_process_local_pdf(pdf_path)
        vectorstore.save_local(store_path)
        with open(hash_path, "w") as f:
            f.write(pdf_hash)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key())
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

    return CompanyChatBot(vectorstore)

chatbot = None  # To store the chatbot instance

async def handle_websocket(websocket):
    global chatbot
    if chatbot is None:
        chatbot = await initialize_chatbot()

    async for message in websocket:
        try:
            print(f"Received message: {message}")  # Debug log
            # Parse the incoming JSON message
            data = json.loads(message)
            user_message = data.get("message", "")
            if not user_message:
                response = {"error": "No message provided."}
            else:
                bot_response = chatbot.process_user_message(user_message)
                response = {"response": bot_response}
        except Exception as e:
            response = {"error": str(e)}

        # Send the response back to the client
        await websocket.send(json.dumps(response))

# Main coroutine to start the WebSocket server
async def main():
    print("WebSocket server started at ws://localhost:8765")
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    # Use asyncio.run to start the event loop and run the main coroutine
    asyncio.run(main())