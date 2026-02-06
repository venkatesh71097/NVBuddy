from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json
import operator

# Import our setup
from lab1_agent.chains import router_chain, nvidia_chain, general_chain
from lab1_agent.mcp_server import _search_nvidia_docs, _calculate_gpu_memory, _sequential_thinking, _search_resume, _simulate_triton_inference, _simulate_tensorrt_build

# --- STATE MANAGEMENT ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    context: str
    resume_context: str = "" # Added explicit field
    generation: str
    chat_history: List[str] # Memory
    # New: Track thoughts for Sequential Thinking
    thought_log: Annotated[List[str], operator.add] 
    step_count: int

# --- NODES ---

def route_question(state: GraphState):
    """Decides which Tutor to call."""
    print("--- 🚦 ROUTING TRAFFIC ---")
    question = state["question"]
    source = router_chain.invoke({"question": question})
    
    # Fallback if the LLM fails to return valid JSON
    if not source:
        print("--- ⚠️ ROUTER FAILED, DEFAULTING TO NVIDIA TUTOR ---")
        return "nvidia_tutor"
    
    # Map pydantic output to graph node names
    if source.datasource == "nvidia_tutor":
        return "nvidia_tutor"
    elif source.datasource == "general_tutor":
        return "general_tutor"
    elif source.datasource == "hardware_calculator":
        return "calc_gpu"
    elif source.datasource == "deployment_expert":
        return "deployment_node"
    return "general_tutor"

def retrieve_docs(state: GraphState):
    """Fetches NVIDIA docs. Only used by NVIDIA Tutor."""
    print("--- 🔍 RETRIEVING NVIDIA DOCS ---")
    question = state["question"]
    docs = _search_nvidia_docs(question) # From MCP
    return {"context": docs}

def thinking_node(state: GraphState):
    """
    The 'Pre-Computation' Loop.
    The agent thinks about how to teach before teaching.
    """
    step = state.get("step_count", 0) + 1
    question = state["question"]
    
    print(f"--- 🧠 THINKING (Step {step}) ---")
    
    # We simulate the tool call here. In a full agent, the LLM would call this itself.
    # For this lab, we force a 3-step planning process:
    # 1. Analyze User Intent
    # 2. Structure the Lesson
    # 3. Simplify/Clarify
    
    thoughts = [
        "Analyzing the user's technical depth and intent...",
        "Structuring the answer: Definition -> Analogy -> Technical Detail...",
        "Reviewing against NVIDIA Best Practices..."
    ]
    
    current_thought = thoughts[min(step-1, 2)]
    
    # Call the MCP Tool to record it
    confirmation = _sequential_thinking(current_thought, needs_more_thought=(step < 3), step=step, total_steps=3)
    
    return {"thought_log": [confirmation], "step_count": step}

def nvidia_tutor_node(state: GraphState):
    print("--- 🟢 NVIDIA TUTOR SPEAKING ---")
    question = state["question"]
    
    # optimize resume search for generic "guide me" queries
    if any(keyword in question.lower() for keyword in ["guide", "start", "roadmap", "plan", "zero"]):
        resume_query = "Technical Skills Experience Projects Architecture"
    else:
        resume_query = question

    resume_context = _search_resume(resume_query)
    
    generation = nvidia_chain.invoke({
        "context": state["context"], 
        "question": question,
        "resume_context": resume_context,
        "chat_history": state.get("chat_history", [])
    })
    return {"generation": generation.content}

def general_tutor_node(state: GraphState):
    print("--- 🔵 GENERAL TUTOR SPEAKING ---")
    generation = general_chain.invoke({
        "question": state["question"],
        "chat_history": state.get("chat_history", [])
    })
    return {"generation": generation.content}

def calc_gpu_node(state: GraphState):
    print("--- 🧮 RUNNING CALCULATOR ---")
    question = state["question"]
    # Simple extraction for Lab 1 (In prod, use an extraction chain)
    # Defaulting to 70B if not found, just for the demo flow
    res = _calculate_gpu_memory(70.0) 
    return {"context": f"HARDWARE CALCULATOR OUTPUT: {res}"}

def deployment_node(state: GraphState):
    print("--- 🚀 DEPLOYMENT EXPERT WORKING ---")
    question = state["question"]
    context_result = ""
    
    # Simple keyword-based tool selection (In prod, use an LLM tool caller)
    if "optimize" in question.lower() or "build" in question.lower() or "engine" in question.lower():
        # User wants TensorRT
        res = _simulate_tensorrt_build("llama3-70b.safetensors", target_precision="FP8")
        context_result = f"TENSORRT BUILDER OUTPUT:\n{res}"
    else:
        # Default to Triton Inference
        res = _simulate_triton_inference("llama-3-70b", batch_size=8)
        context_result = f"TRITON SERVER OUTPUT:\n{res}"
        
    return {"context": context_result}

# --- GRAPH ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("think", thinking_node)
workflow.add_node("nvidia_tutor", nvidia_tutor_node)
workflow.add_node("general_tutor", general_tutor_node)
workflow.add_node("calc_gpu", calc_gpu_node)
workflow.add_node("deployment_node", deployment_node)

# LOGIC:
# 1. Start -> Route
# 2. Route -> NVIDIA (Retrieve -> Think -> Speak)
# 3. Route -> General (Think -> Speak)
# 4. Route -> Calc (Calc -> NVIDIA Tutor)  <-- CHANGED: Calculator feeds the Tutor

def decide_entry(state: GraphState):
    target = route_question(state)
    if target == "nvidia_tutor":
        return "retrieve" # NVIDIA needs docs first
    elif target == "general_tutor":
        return "think"    # General goes straight to thinking
    elif target == "calc_gpu":
        return "calc_gpu" # Calculate first
    elif target == "deployment_node":
        return "deployment_node"
    return "general_tutor"

workflow.set_conditional_entry_point(
    decide_entry,
    {
        "retrieve": "retrieve",
        "think": "think",
        "calc_gpu": "calc_gpu",
        "deployment_node": "deployment_node"
    }
)

# Edges
workflow.add_edge("retrieve", "think")
workflow.add_edge("calc_gpu", "nvidia_tutor") # Calculator result -> Tutor Explain
workflow.add_edge("deployment_node", "nvidia_tutor") # Deployment Result -> Tutor Explain

# Edges for Thinking Loop

# Edges for Thinking Loop
def check_thought_process(state: GraphState):
    if state["step_count"] < 3:
        return "loop"
    
    # If done thinking, go to the right tutor
    # We need to know WHICH tutor we were aiming for.
    # A cleaner way: The 'Think' node creates the plan, but who executes?
    # Hack for Lab 1: We'll re-route based on the context existence.
    if state.get("context"): 
        return "nvidia"
    else:
        return "general"

workflow.add_conditional_edges(
    "think",
    check_thought_process,
    {
        "loop": "think",
        "nvidia": "nvidia_tutor",
        "general": "general_tutor"
    }
)

workflow.add_edge("nvidia_tutor", END)
workflow.add_edge("general_tutor", END)
workflow.add_edge("calc_gpu", END)

app = workflow.compile()
