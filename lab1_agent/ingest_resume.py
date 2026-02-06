import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RESUME_PATH = "resume.pdf"
INDEX_PATH = "lab1_agent/resume_cv_index"

def ingest_resume():
    print("--- 📄 INGESTING RESUME ---")
    
    if not os.path.exists(RESUME_PATH):
        print(f"ERROR: {RESUME_PATH} not found. Please place your PDF in the root folder.")
        return

    loader = PyPDFLoader(RESUME_PATH)
    documents = loader.load()
    print(f"Loaded Resume: {len(documents)} pages.")

    # Split resume into smaller chunks to find specific skills
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(documents)

    if not os.getenv("NVIDIA_API_KEY"):
         print("ERROR: NVIDIA_API_KEY missing.")
         return

    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    
    # Separate Index for Resume (don't mix with H100 docs)
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"--- SUCCESS: Resume Index saved to '{INDEX_PATH}' ---")

if __name__ == "__main__":
    ingest_resume()
