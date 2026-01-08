from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query import setup_chain, chat_step, chat_step_async, clear_caches
from typing import List, Optional
import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import time

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

# ============= PERFORMANCE OPTIMIZATIONS =============
# 1. Pre-initialize chain on startup (warmup)
# 2. Reuse chain instances across requests
# 3. Async processing for non-blocking responses
# 4. Connection pooling and timeouts
# 5. Request timeout handling

# Global chain - initialized once on startup
_global_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    global _global_chain
    
    # Startup: Pre-initialize chain for faster first request
    logger.info("🚀 Pre-initializing chatbot chain...")
    start_time = time.time()
    
    try:
        _global_chain = setup_chain()
        elapsed = time.time() - start_time
        logger.info(f"✅ Chain initialized in {elapsed:.2f}s - Ready to serve requests!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chain: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down chatbot API...")
    clear_caches()

app = FastAPI(
    title="Chatbot API",
    description="High-performance API endpoint for chatbot queries",
    lifespan=lifespan
)

# Configure allowed origins from .env file
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,*")
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

# Response model
class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    sources: Optional[List[dict]] = None

# Store chains per conversation (limited for memory efficiency)
MAX_CONVERSATIONS = 100  # Limit to prevent memory bloat
conversation_chains = {}
conversation_last_used = {}

def cleanup_old_conversations():
    """Remove old conversations to free memory."""
    global conversation_chains, conversation_last_used
    
    if len(conversation_chains) > MAX_CONVERSATIONS:
        # Sort by last used time and remove oldest half
        sorted_convos = sorted(conversation_last_used.items(), key=lambda x: x[1])
        to_remove = sorted_convos[:len(sorted_convos)//2]
        
        for conv_id, _ in to_remove:
            if conv_id in conversation_chains:
                del conversation_chains[conv_id]
            if conv_id in conversation_last_used:
                del conversation_last_used[conv_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old conversations")

def get_or_create_chain(conversation_id: str = "default"):
    """Get or create a chain for a conversation - optimized."""
    global _global_chain
    
    # Use global chain for default conversations (fastest)
    if conversation_id == "default" and _global_chain:
        return _global_chain
    
    # Check if we have a cached chain for this conversation
    if conversation_id not in conversation_chains:
        # Cleanup if too many conversations
        cleanup_old_conversations()
        conversation_chains[conversation_id] = setup_chain()
    
    # Update last used time
    conversation_last_used[conversation_id] = time.time()
    
    return conversation_chains[conversation_id]

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Chatbot API is running",
        "chain_ready": _global_chain is not None,
        "active_conversations": len(conversation_chains),
        "allowed_origins": ALLOWED_ORIGINS
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chain_initialized": _global_chain is not None,
        "active_conversations": len(conversation_chains)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - optimized for speed.
    Uses pre-initialized chain for fastest response.
    """
    start_time = time.time()
    conversation_id = request.conversation_id or "default"
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"[{conversation_id}] Received query: {query[:100]}...")
    
    try:
        # Get chain (uses cached/pre-initialized chain)
        chain = get_or_create_chain(conversation_id)
        
        # Process the query using async version for non-blocking
        response, history = await chat_step_async(chain, query)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[{conversation_id}] Response in {response_time_ms}ms")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            response_time_ms=response_time_ms
        )
    except asyncio.TimeoutError:
        logger.error(f"[{conversation_id}] Request timeout for query: {query[:50]}...")
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        logger.error(f"[{conversation_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """
    Synchronous chat endpoint - for compatibility.
    Slightly faster for single requests but blocks the event loop.
    """
    start_time = time.time()
    conversation_id = request.conversation_id or "default"
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        chain = get_or_create_chain(conversation_id)
        response, history = chat_step(chain, query)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            response_time_ms=response_time_ms
        )
    except Exception as e:
        logger.error(f"[{conversation_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/clear")
async def clear_chat(conversation_id: str = "default"):
    """Clear chat history for a conversation."""
    try:
        if conversation_id in conversation_chains:
            conversation_chains[conversation_id].memory.clear()
            return {"status": "success", "message": f"Chat history cleared for {conversation_id}"}
        elif conversation_id == "default" and _global_chain:
            _global_chain.memory.clear()
            return {"status": "success", "message": "Default chat history cleared"}
        else:
            return {"status": "success", "message": f"No conversation found for {conversation_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat: {str(e)}")

@app.post("/refresh")
async def refresh_chain(background_tasks: BackgroundTasks):
    """Refresh the chatbot chain (reload data from Pinecone)."""
    global _global_chain
    
    def do_refresh():
        global _global_chain
        clear_caches()
        _global_chain = setup_chain(force_new=True)
        logger.info("Chain refreshed successfully")
    
    background_tasks.add_task(do_refresh)
    return {"status": "refreshing", "message": "Chain refresh started in background"}

@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "active_conversations": len(conversation_chains),
        "max_conversations": MAX_CONVERSATIONS,
        "chain_ready": _global_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run with optimized settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for shared chain state
        timeout_keep_alive=30,
        access_log=False  # Disable access log for speed
    )
