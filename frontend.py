import streamlit as st
import websockets
import asyncio
import json

async def send_message(message):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"message": message}))
        response = await websocket.recv()
        return json.loads(response)

st.title("Agentic RAG Chatbot")
st.write("Ask questions about the company profile, request it via email, or just chat!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_message = st.text_input("Enter your message", "")

if st.button("Send") and user_message:
    st.session_state["messages"].append(("You", user_message))
    response = asyncio.run(send_message(user_message))
    bot_response = response.get("response", "Sorry, there was an error.")
    st.session_state["messages"].append(("Bot", bot_response))

for role, msg in st.session_state["messages"]:
    st.write(f"**{role}:** {msg}")
