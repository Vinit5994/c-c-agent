"""
Example of how to call the API endpoint
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# API endpoint URL from .env file
# Format in .env: API_URL=http://localhost:8000
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Example 1: Simple chat request
def chat_example():
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "query": "What is this website about?",
            "conversation_id": "user123"  # Optional: for managing multiple conversations
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("Response:", data["response"])
        print("Conversation ID:", data.get("conversation_id"))
    else:
        print("Error:", response.status_code, response.text)

# Example 2: Clear chat history
def clear_chat_example():
    response = requests.post(
        f"{API_URL}/chat/clear",
        params={"conversation_id": "user123"}
    )
    
    if response.status_code == 200:
        print("Chat cleared:", response.json())
    else:
        print("Error:", response.status_code, response.text)

# Example 3: Health check
def health_check():
    response = requests.get(f"{API_URL}/")
    print("Status:", response.json())

if __name__ == "__main__":
    # Make sure the API is running first!
    # Run: python api.py or uvicorn api:app --reload
    
    print("=== Health Check ===")
    health_check()
    
    print("\n=== Chat Example ===")
    chat_example()
    
    print("\n=== Clear Chat Example ===")
    clear_chat_example()

