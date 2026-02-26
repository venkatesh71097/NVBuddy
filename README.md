# 🤖 NVBuddy (v2.0 Redesign)

![NVIDIA Solutions Architect](https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg)

An AI-powered "Study Buddy" designed to help candidates prepare for NVIDIA Solutions Architect (SA) interviews. This tool leverages a **Dual-Brain Agent** architecture to simulate technical deep dives, system design discussions, and hardware sizing calculations. 

*Note: Originally built with Streamlit, this project was redesigned into a production-grade React+Vite web app with a FastAPI backend to support advanced UI cards and Vercel deployment.*

---

## 🧠 Core Features

### 1. Dual-Brain Architecture
The agent intelligently routes your queries to the most appropriate "expert":
- **🟢 NVIDIA Tutor (Technical)**: A RAG-enabled expert on NVIDIA technologies (NeMo, H100, Triton, etc.). It adopts a "Skeptical CTO" persona to challenge your understanding and relates concepts directly to your resume.
- **🔵 General Tutor (System Design)**: A Socratic interviewer for general coding and system design questions. It helps you structure your answers using frameworks like STAR or RESHAPED.
- **🧮 Hardware Calculator**: Automatically handles complex VRAM and GPU sizing math (e.g., "How many H100s for Llama 3 70B?").
- **🚀 Deployment Expert**: Simulates scenarios for optimizing inference with Triton Inference Server and TensorRT-LLM.

### 2. Sequential Thinking
Just like a real candidate, the agent uses a "Check-Plan-Act" loop to structure its thoughts before responding, ensuring high-quality, structured answers.

### 3. Resume Integration
The agent ingests your `resume.pdf` to provide personalized context. If you ask "Where do I start?", it identifies gaps in your experience relative to NVIDIA's requirements.

---

## 📸 Interaction Demo

Watch the Dual-Brain Agent handle a technical question about Triton Inference Server:

https://github.com/venkatesh71097/NVBuddy/raw/main/demo.mp4

---

## 🛠️ Technology Stack

- **Frontend**: React, TypeScript, Vite, Lucide Icons (Premium NVIDIA Dark Theme)
- **Backend API**: FastAPI (Python)
- **Agent Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful multi-actor orchestration)
- **LLM**: Llama-3.1-70B via [NVIDIA AI Endpoints](https://build.nvidia.com/explore/discover)
- **Vector Store**: FAISS (for NVIDIA Docs and Resume RAG)
- **Tools**: Custom MCP-style tools for Sequential Thinking and Math

---

## 📂 Project Structure

```bash
├── frontend/               # React + Vite Frontend
│   ├── src/
│   │   ├── App.tsx         # Main UI Layout
│   │   ├── index.css       # NVIDIA Design System
│   │   ├── components/     # UI Components (ChatPanel)
│   │   └── services/api.ts # FastAPI Client
├── server.py               # FastAPI Backend Entry Point
├── lab1_agent/             # Core Agent Logic
│   ├── graph.py            # LangGraph Workflow Definition
│   ├── chains.py           # LangChain Prompts & Router
│   ├── mcp_server.py       # Tools (Search, Math, Simulation)
│   └── ingest.py           # RAG Data Ingestion Scripts
├── requirements.txt        # Python Dependencies
└── resume.pdf              # Your Resume (Context Source)
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- Node.js (for frontend)
- Python 3.10+ (for backend)
- An NVIDIA AI Foundation Models API Key

### 1. Backend Setup

```bash
# Clone the repository
git clone <repo-url>
cd "NVIDIA Solutions Architect Interview Prep"

# Install Python dependencies
pip install -r requirements.txt

# Set your API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env

# Data Ingestion (First run only)
# Make sure resume.pdf is in the root directory
python lab1_agent/ingest.py
python lab1_agent/ingest_resume.py

# Start the FastAPI Server (Port 8000)
uvicorn server:app --reload
```

### 2. Frontend Setup

Open a new terminal window:

```bash
cd frontend
npm install

# Start the Vite Dev Server (Port 8502)
npm run dev -- --port 8502
```

Navigate to `http://localhost:8502` to use the app.

---

## 🌐 Deployment Architecture

Since the frontend is a static React app and the backend relies on persistent memory (FAISS Vector Store), they must be deployed separately:

1. **Frontend**: Deploy to **Vercel** or **Netlify**. Ensure the API URL in `api.ts` points to your backend production URL.
2. **Backend**: Deploy to **Render** or **Heroku**. These platforms support persistent background memory, which is required to hold the FAISS vector index in RAM. Serverless functions (like Vercel API routes) will constantly drop the vector index and crash.

---

*Disclaimer: This is a study tool and not an official NVIDIA product.*
