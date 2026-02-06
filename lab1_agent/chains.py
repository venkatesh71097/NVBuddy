from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import os
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0, api_key='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')

# --- 1. THE ROUTER (Traffic Control) ---
class RouteQuery(BaseModel):
    """Route a user query to the correct expert tutor."""
    datasource: Literal["nvidia_tutor", "general_tutor", "hardware_calculator", "deployment_expert"] = Field(
        ...,
        description="Route to 'nvidia_tutor' for NeMo/H100/Concept questions. Route to 'deployment_expert' for Triton/TensorRT/Inference/Engine building questions. Route to 'general_tutor' for coding rules. Route to 'hardware_calculator' for VRAM math."
    )

router_system_prompt = """You are a Technical Learning Coordinator.
Route the user's question to the expert:
1. 'nvidia_tutor': For general NVIDIA concepts (NeMo, Guardrails, H100 architecture).
2. 'deployment_expert': SPECIFICALLY for Model Deployment, Inference, Triton Server, TensorRT-LLM, Engine Building, or Latency Optimization.
3. 'general_tutor': For generic coding or system design.
4. 'hardware_calculator': For VRAM math.
5. If in doubt, default to 'nvidia_tutor'.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{question}"),
])

router_chain = router_prompt | llm.with_structured_output(RouteQuery)


# --- 2. NVIDIA TUTOR (RAG-Based) ---
nvidia_tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior NVIDIA Solutions Architect acting as a mentor, AND sometimes a Skeptical CTO.
    
    Your goal is to prepare the candidate for a grilling.
    
    CANDIDATE CONTEXT (RESUME):
    {resume_context}
    
    INSTRUCTIONS:
    1. Answer the question using ONLY the provided context.
    2. RELATE IT TO THEIR RESUME: If they ask about H100s, and their resume says "Financial Modeling", explain how H100s speed up Monte Carlo simulations.
    3. BEAST MODE (The CTO): Occassionally challenge them. "Are you sure? Why not just use AWS?"
    4. Use NVIDIA-specific terminology (e.g., don't just say "fast connection", say "NVLink").
    5. PROACTIVE MODE: If the user asks "Guide me" or "Where to start":
       - Look at their RESUME strengths/weaknesses.
       - Pick a Critical NVIDIA Topic (e.g., Triton, NeMo, CUDA).
       - Say: "I see you know X, but do you know Y? Let's start there."
       - Then TEACH it immediately. Don't ask "what do you want?".
    """),
    ("placeholder", "{chat_history}"),
    ("human", "Context: {context} \n\n Question: {question}"),
])
nvidia_chain = nvidia_tutor_prompt | llm


# --- 3. GENERAL TUTOR (Reasoning-Based) ---
general_tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior System Design & Algorithm Interviewer.
    Your goal is to help the candidate pass the "General Technical" rounds.
    
    INSTRUCTIONS:
    1. Do NOT give the answer immediately. Be Socratic.
    2. Suggest a framework (e.g., STAR for behavioral, RESHAPED for System Design).
    3. If it's a coding question, suggest a pattern (e.g., "Two Pointers", "DFS") before showing code.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])
general_chain = general_tutor_prompt | llm
