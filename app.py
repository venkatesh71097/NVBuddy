import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Page Config (NVIDIA Colors)
st.set_page_config(
    page_title="NVIDIA AI Study Buddy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NVIDIA feel
st.markdown("""
<style>
    .stChatInputContainer {
        border-color: #76b900 !important;
    }
    h1 {
        color: #76b900 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=150)
    st.title("Dual-Brain Agent")
    st.markdown("---")
    
    # Status Indicators
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        st.success("✅ NVIDIA RAM (API Key)")
    else:
        st.error("❌ No API Key Found")
        
    st.markdown("### 🧠 Brain Status")
    if os.path.exists("lab1_agent/nvidia_faiss_index"):
        st.success("✅ Technical Knowledge Base")
    else:
        st.warning("⚠️ technical KB missing")
        
    if os.path.exists("lab1_agent/resume_cv_index"):
        st.success("✅ Resume Context")
    else:
        st.warning("⚠️ Resume Context missing")
    
    st.markdown("---")
    st.markdown("**Modes:**")
    st.info("1. NVIDIA Tutor (Technical)")
    st.info("2. General Tutor (System Design)")
    st.info("3. Hardware Calculator (Sizing)")

# Main Chat Interface
st.title("🤖 NVIDIA Solutions Architect Prep")
st.caption("Powered by Llama-3.1-70B, RAG, and Sequential Thinking")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Sync Wrapper for LangGraph
# We use .stream() (Sync) instead of .astream() to avoid Event Loop issues in Streamlit
def run_agent(user_input):
    from lab1_agent.graph import app as agent_app
    
    # Placeholder for the answer
    assistant_msg = st.chat_message("assistant")
    response_placeholder = assistant_msg.empty()
    
    # Status container for "Thinking"
    status_container = assistant_msg.status("🧠 Agent is thinking...", expanded=True)
    
    final_answer = ""
    
    try:
        # Convert Strealit history to LangChain format (Simple list of strings for Llama 3)
        history = []
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history.append(f"{role}: {msg['content']}")

        inputs = {
            "question": user_input,
            "chat_history": history[-4:] # Keep only last 4 messages to prevent context bloat/hanging
        }
        
        # Run Synchronously
        status_container.write("🔄 Connecting to Brain...")
        for output in agent_app.stream(inputs, stream_mode="updates"):
            status_container.write(f"📡 Receiving Signal: {list(output.keys())[0]}")
            for key, value in output.items():
                # Update the Thinking Status
                if key == "retrieve":
                    status_container.write("🔍 Retrieving NVIDIA Docs...")
                elif key == "think":
                    thought = value.get("thought_log", ["Thinking..."])[-1]
                    status_container.write(f"🤔 Thought: {thought}")
                elif key == "calc_gpu":
                    status_container.write("🧮 Running Hardware Calculator...")
                elif key in ["nvidia_tutor", "general_tutor"]:
                    # This is the final answer
                    final_answer = value["generation"]
    
        # Final UI Updates
        status_container.update(label="✅ Thought Process Complete", state="complete", expanded=False)
        response_placeholder.markdown(final_answer)
        
        return final_answer
        
    except Exception as e:
        status_container.update(label="❌ Error", state="error")
        st.error(f"Agent Crashed: {str(e)}")
        return None

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'Quiz me on NeMo based on my resume')"):
    # Add User Message to Chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Run Agent
    answer = run_agent(prompt)
    
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
