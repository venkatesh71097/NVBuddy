from fastmcp import FastMCP
from typing import Dict, List, Optional
import json

# Create the MCP Server
# "NVIDIA_Prep_Server" is the name of our tool provider
mcp = FastMCP("NVIDIA_Prep_Server")

# --- Theory Interspersed ---
# INTERVIEW TIP:
# The "Model Context Protocol" (MCP) is likely a key interview topic because NVIDIA
# mentions it in the JD. It solves the "N*M" problem:
# Instead of building N integrations for M agents, you build 1 MCP Server
# that any MCP Client (Claude, LangChain Agent) can use.

from lab1_agent.tools.triton_mock import _simulate_triton_inference
from lab1_agent.tools.tensorrt_mock import _simulate_tensorrt_build

from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# --- REAL IMPLEMENTATION ---
# Load the Vector Database once on startup
# This is how production microservices work (load weights/index in memory)

VECTOR_DB_PATH = "lab1_agent/nvidia_faiss_index"
vector_store = None

try:
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Loading FAISS index from {VECTOR_DB_PATH}...")
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
        # Allow dangerous deserialization because we created the file ourselves
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Index loaded successfully.")
    else:
        print("WARNING: FAISS index not found. Please run 'python lab1_agent/ingest.py' first.")

    RESUME_DB_PATH = "lab1_agent/resume_cv_index"
    resume_store = None
    if os.path.exists(RESUME_DB_PATH):
        print(f"Loading Resume index from {RESUME_DB_PATH}...")
        resume_store = FAISS.load_local(RESUME_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Resume Index loaded.")
except Exception as e:
    print(f"Error loading index: {e}")

# --- CORE LOGIC (Raw Functions) ---
def _search_nvidia_docs(query: str) -> str:
    """Internal search function."""
    if not vector_store:
        return "System Error: Vector Database not loaded."
    
    results = vector_store.similarity_search(query, k=3)
    context_str = ""
    for i, doc in enumerate(results):
        context_str += f"\n[Result {i+1} Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}\n"
    return context_str

def _search_resume(query: str) -> str:
    """Internal search function for resume."""
    if not resume_store:
        return "No resume loaded."
    
    results = resume_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

@mcp.tool()
def search_nvidia_docs(query: str) -> str:
    """Search NVIDIA technical documentation for specific concepts using Semantic Search."""
    return _search_nvidia_docs(query)


def _sequential_thinking(thought: str, needs_more_thought: bool, step: int, total_steps: int) -> str:
    return f"Thought recorded: {thought}. Step {step}/{total_steps}."

@mcp.tool()
def sequential_thinking(thought: str, needs_more_thought: bool, step: int, total_steps: int) -> str:
    """Use this tool to plan your teaching strategy."""
    return _sequential_thinking(thought, needs_more_thought, step, total_steps)

def _calculate_gpu_memory(params_billion: float, precision: str = "FP16", context_window: int = 4096) -> str:
    bytes_per_param = {
        "FP16": 2, "BF16": 2, "FP8": 1, "INT8": 1, "INT4": 0.5
    }.get(precision.upper(), 2)
    
    weight_vram_gb = params_billion * bytes_per_param
    total_vram_gb = weight_vram_gb * 1.2
    
    gpu_options = {"H100": 80, "A100": 80, "L40S": 48}
    
    recommendation = []
    for gpu, vram in gpu_options.items():
        count = (-(-total_vram_gb // vram)) # ceil division
        recommendation.append(f"{int(count)}x {gpu} ({vram}GB)")
        
    return json.dumps({
        "estimated_model_size_gb": f"{weight_vram_gb:.1f} GB",
        "total_vram_with_overhead": f"{total_vram_gb:.1f} GB",
        "recommended_gpus": recommendation,
        "formula_used": f"{params_billion}B params * {bytes_per_param} bytes/param * 1.2 overhead"
    }, indent=2)

@mcp.tool()
def calculate_gpu_memory(params_billion: float, precision: str = "FP16", context_window: int = 4096) -> str:
    """Calculate estimated VRAM usage for an LLM."""
    return _calculate_gpu_memory(params_billion, precision, context_window)

if __name__ == "__main__":
    # This starts the MCP server on stdio, listening for client requests
    mcp.run()
