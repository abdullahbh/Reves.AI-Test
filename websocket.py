import asyncio
import websockets
import json
import os

from bot import initialize_agentic_bot, CompanyAgentBot

agentic_bot = None  # We'll store our agent wrapper here

async def initialize_bot():
    global agentic_bot
    if agentic_bot is None:
        # Adjust if your PDF name/path differs
        agent = initialize_agentic_bot(pdf_path="company_profile.pdf", store_path="my_vectorstore")
        agentic_bot = CompanyAgentBot(agent)
    return agentic_bot

async def handle_websocket(websocket):
    await initialize_bot()  # Ensure the agent is ready

    async for message in websocket:
        try:
            data = json.loads(message)
            user_message = data.get("message", "")
            if not user_message:
                response = {"error": "No message provided."}
            else:
                response_text = agentic_bot.process_message(user_message)
                response = {"response": response_text}
        except Exception as e:
            response = {"error": str(e)}

        await websocket.send(json.dumps(response))

async def main():
    print("WebSocket server started at ws://localhost:8765")
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
