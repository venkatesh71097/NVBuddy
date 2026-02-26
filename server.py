from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from lab1_agent.graph import app as agent_app

app = FastAPI(title="NVBuddy API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In prod, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = []

@app.get("/api/status")
def get_status():
    import os
    return {
        "status": "online",
        "api_key_set": bool(os.getenv("NVIDIA_API_KEY")),
        "faiss_index_exists": os.path.exists("lab1_agent/nvidia_faiss_index"),
        "resume_index_exists": os.path.exists("lab1_agent/resume_cv_index")
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Convert incoming history to format expected by the agent
    history = []
    for msg in req.chat_history[-4:]: # Keep last 4 for context
        role = "User" if msg.role == "user" else "Assistant"
        history.append(f"{role}: {msg.content}")

    inputs = {
        "question": req.question,
        "chat_history": history
    }
    
    final_answer = ""
    target_node = ""
    thought_log = []
    
    # We iterate over the stream to gather thoughts and the final answer
    async for output in agent_app.astream(inputs, stream_mode="updates"):
        for key, value in output.items():
            if key == "think":
                if "thought_log" in value:
                    thought_log.append(value["thought_log"][-1])
            elif key in ["nvidia_tutor", "general_tutor"]:
                final_answer = value.get("generation", "")
                target_node = key
                
    return {
        "answer": final_answer,
        "node": target_node,
        "thoughts": thought_log
    }

if __name__ == "__main__":
    print("Starting NVBuddy FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
