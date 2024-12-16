import streamlit as st
import websockets
import asyncio
import json

# Function to interact with WebSocket
async def send_message(message):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"message": message}))
        response = await websocket.recv()
        return json.loads(response)

# Streamlit UI
st.title("RAG Chatbot")
st.write("Ask questions about the company profile.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_message = st.text_input("Enter your question", "")

if st.button("Send") and user_message:
    st.session_state["messages"].append(("user", user_message))
    response = asyncio.run(send_message(user_message))
    bot_response = response.get("response", "Sorry, there was an error.")
    st.session_state["messages"].append(("bot", bot_response))

# Display chat history
for role, message in st.session_state["messages"]:
    if role == "user":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")
