import os
import requests
from dotenv import load_dotenv

# Load env variables
load_dotenv()
key = os.getenv("NVIDIA_API_KEY")

if not key:
    print("ERROR: No NVIDIA_API_KEY found in .env")
    exit(1)

print(f"--- TESTING API KEY: {key[:10]}... ---")

# Endpoint for Llama 3 70B
url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.text)
    
    if response.status_code == 200:
        print("\nSUCCESS: Key is working!")
    elif response.status_code == 401:
        print("\nFAILURE: 401 Unauthorized. Your key is invalid, expired, or has no credits.")
    elif response.status_code == 402:
        print("\nFAILURE: 402 Payment Required. You need to add credits.")
        
except Exception as e:
    print(f"Network Error: {e}")
