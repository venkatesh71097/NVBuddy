import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import time

load_dotenv()

# --- THEORY: "Document Loaders" ---
# In production, we don't hardcode text. We use "Loaders" to fetch data from
# sources like Web URLs, S3 Buckets, Google Drive, or Slack.
# Here, we use 'WebBaseLoader' to scrape the live NVIDIA documentation.

TARGET_URLS = [
    # 1. H100 Architecture (The "Hardware" Bible)
    "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
    
    # 2. Triton Inference Server (The "Deployment" Bible)
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html",
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html",
    
    # 3. NeMo Framework (The "Training" Bible)
    "https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html",
    "https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron.html",
    
    # 4. Agentic AI (The "Modern" Stuff)
    "https://developer.nvidia.com/blog/building-your-first-llm-agent-application/",
    # 4. Agentic AI (The "Modern" Stuff)
    "https://developer.nvidia.com/blog/building-your-first-llm-agent-application/",
]

def _crawl_nvidia_blogs(category_url: str, max_pages: int = 2):
    """
    Crawls NVIDIA Blog Category to find article URLs.
    """
    print(f"🕷️ Crawling {category_url} for {max_pages} pages...")
    found_urls = []
    
    for page in range(1, max_pages + 1):
        target = f"{category_url}page/{page}" if page > 1 else category_url
        try:
            print(f"   Fetching: {target}")
            resp = requests.get(target, timeout=10)
            if resp.status_code != 200:
                print(f"   Failed to fetch page {page}: {resp.status_code}")
                continue
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Look for article links (NVIDIA Blog structure)
            # Typically h3 class="entry-title" -> a href
            # Or general link finding within the main list
            
            # Heuristic: Find all links that are in the /blog/ path and avoiding categories/tags
            links = soup.select("h3.entry-title a")
            
            for link in links:
                href = link.get('href')
                if href and "/blog/" in href:
                    found_urls.append(href)
                    
            print(f"   Found {len(links)} articles on page {page}.")
            time.sleep(1) # Be polite
            
        except Exception as e:
            print(f"   Error crawling page {page}: {e}")
            
    # Deduplicate
    return list(set(found_urls))

def ingest_data():
    print("--- 1. LOADING DATA (LIVE WEB SCRAPING) ---")
    if not os.getenv("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY is missing. Please set it in .env")
        return

    # WebBaseLoader uses BeautifulSoup to parse HTML into text
    # WebBaseLoader uses BeautifulSoup to parse HTML into text
    
    # CRAWL: Generative AI Blog
    blog_urls = _crawl_nvidia_blogs("https://developer.nvidia.com/blog/category/generative-ai/", max_pages=3)
    
    all_urls = TARGET_URLS + blog_urls
    print(f"Total URLs to Ingest: {len(all_urls)} (Base: {len(TARGET_URLS)}, Blog: {len(blog_urls)})")
    
    loader = WebBaseLoader(all_urls)
    print(f"Scraping... (This may take a moment)")
    
    try:
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages.")
    except Exception as e:
        print(f"Error scraping data: {e}")
        return

    print("--- 2. CHUNKING (TEXT SPLITTING) ---")
    # Real web pages are huge. We typically use a larger chunk size.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = splitter.split_documents(documents)
    print(f"Split into {len(chunked_docs)} chunks.")

    print("--- 3. EMBEDDING & INDEXING (FAISS) ---")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    
    save_path = "lab1_agent/nvidia_faiss_index"
    vectorstore.save_local(save_path)
    print(f"--- SUCCESS: Index saved to '{save_path}' ---")
    print("The Agent now has access to the LIVE content from these websites.")

if __name__ == "__main__":
    ingest_data()
