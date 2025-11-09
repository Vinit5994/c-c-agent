from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import setup_chain, chat_step
from typing import List, Optional
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API", description="API endpoint for chatbot queries")

# Configure allowed origins from .env file
# Format in .env: ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000,https://yourdomain.com
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Only allow specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allowed HTTP methods
    allow_headers=["*"],  # Allowed headers
)

# Request model
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None  # Optional: for managing multiple conversations

# Response model
class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    sources: Optional[List[dict]] = None

# Store chains per conversation (in production, use Redis or database)
conversation_chains = {}

def get_or_create_chain(conversation_id: str = "default"):
    """Get or create a chain for a conversation."""
    if conversation_id not in conversation_chains:
        conversation_chains[conversation_id] = setup_chain()
    return conversation_chains[conversation_id]

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Chatbot API is running",
        "allowed_origins": ALLOWED_ORIGINS
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes queries and returns responses.
    Only accessible from allowed origins.
    """
    conversation_id = request.conversation_id or "default"
    query = request.query
    
    # Log incoming request
    logger.info(f"[{conversation_id}] Received query: {query}")
    
    try:
        # Get or create chain for this conversation
        chain = get_or_create_chain(conversation_id)
        
        # Process the query
        response, history = chat_step(chain, query)
        
        # Log response
        logger.info(f"[{conversation_id}] Response length: {len(response)} chars")
        logger.info(f"[{conversation_id}] Response preview: {response[:200]}...")
        
        # Log if response indicates no information
        if "don't have enough information" in response.lower():
            logger.warning(f"[{conversation_id}] ⚠️ Response indicates insufficient information for query: {query}")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id
        )
    except Exception as e:
        logger.error(f"[{conversation_id}] Error processing query '{query}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/chat/clear")
async def clear_chat(conversation_id: str = "default"):
    """Clear chat history for a conversation."""
    try:
        if conversation_id in conversation_chains:
            conversation_chains[conversation_id].memory.clear()
            return {"status": "success", "message": f"Chat history cleared for conversation {conversation_id}"}
        else:
            return {"status": "success", "message": f"No conversation found for {conversation_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

