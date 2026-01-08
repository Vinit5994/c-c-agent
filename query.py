import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
from urllib.parse import urlparse
import time

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
INDEX_NAME = "cc"

# ============= PERFORMANCE OPTIMIZATIONS =============
# 1. Use faster model with lower max_tokens for speed
# 2. Disable hallucination check (saves 50% latency)
# 3. Cache embeddings and namespace lookups
# 4. Parallel namespace search
# 5. Reduced retrieval count (k=6 instead of 12)
ENABLE_HALLUCINATION_CHECK = False  # Set True only if accuracy is critical
RETRIEVAL_K = 6  # Reduced from 12 - faster retrieval
MAX_CONTEXT_CHARS = 3000  # Limit context size for faster processing

# Single LLM instance (reused across calls)
_llm_instance = None
_embeddings_instance = None
_pinecone_instance = None
_namespace_cache = {}
_namespace_cache_time = 0
NAMESPACE_CACHE_TTL = 300  # 5 minutes cache

def get_llm():
    """Get cached LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=500,  # Limit output tokens for speed
            request_timeout=30  # Timeout for reliability
        )
    return _llm_instance

def get_embeddings():
    """Get cached embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=384
        )
    return _embeddings_instance

def get_pinecone():
    """Get cached Pinecone instance."""
    global _pinecone_instance
    if _pinecone_instance is None:
        _pinecone_instance = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return _pinecone_instance

def get_site_name_from_sources(sources):
    """Extract site name from source documents."""
    if sources and len(sources) > 0:
        url = sources[0].metadata.get('url', '')
        if url:
            parsed = urlparse(url)
            site_name = parsed.netloc.replace('www.', '')
            return site_name
    return "the website"

def get_all_namespaces():
    """Get all namespaces from Pinecone index with caching."""
    global _namespace_cache, _namespace_cache_time
    
    current_time = time.time()
    
    # Return cached namespaces if still valid
    if _namespace_cache and (current_time - _namespace_cache_time) < NAMESPACE_CACHE_TTL:
        logger.info(f"Using cached namespaces: {_namespace_cache}")
        return _namespace_cache
    
    try:
        pc = get_pinecone()
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        if '' not in namespaces and stats.get('total_vector_count', 0) > 0:
            namespaces.append('')
        
        # Update cache
        _namespace_cache = namespaces if namespaces else [None]
        _namespace_cache_time = current_time
        
        logger.info(f"Fetched {len(_namespace_cache)} namespaces: {_namespace_cache}")
        return _namespace_cache
    except Exception as e:
        logger.warning(f"Could not get namespaces: {e}. Using default namespace.")
        return [None]

class FastMultiNamespaceRetriever(BaseRetriever):
    """Optimized retriever with parallel namespace search."""
    
    index_name: str
    embeddings: OpenAIEmbeddings
    namespaces: List[str]
    k: int = RETRIEVAL_K
    vectorstores: dict = {}
    _executor: Optional[ThreadPoolExecutor] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, index_name, embeddings, namespaces, k=RETRIEVAL_K):
        vectorstores_dict = {}
        for ns in namespaces:
            vectorstores_dict[ns] = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                namespace=ns if ns else None
            )
        
        super().__init__(
            index_name=index_name,
            embeddings=embeddings,
            namespaces=namespaces,
            k=k,
            vectorstores=vectorstores_dict
        )
        self._executor = ThreadPoolExecutor(max_workers=min(len(namespaces), 4))
        logger.info(f"Initialized FastMultiNamespaceRetriever with {len(self.vectorstores)} namespaces")
    
    def _search_single_namespace(self, ns_vectorstore_tuple, query: str):
        """Search a single namespace - for parallel execution."""
        ns, vectorstore = ns_vectorstore_tuple
        try:
            results_with_scores = vectorstore.similarity_search_with_score(query, k=self.k)
            return [(doc, score, ns) for doc, score in results_with_scores]
        except Exception as e:
            logger.warning(f"Error searching namespace '{ns or 'default'}': {e}")
            return []
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search across all namespaces IN PARALLEL and return top k results."""
        start_time = time.time()
        
        # Parallel search across all namespaces
        futures = []
        for ns, vectorstore in self.vectorstores.items():
            future = self._executor.submit(
                self._search_single_namespace, 
                (ns, vectorstore), 
                query
            )
            futures.append(future)
        
        # Collect all results
        all_results = []
        for future in futures:
            try:
                results = future.result(timeout=10)  # 10 second timeout per namespace
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Timeout or error in namespace search: {e}")
        
        # Sort by score (lower is better for cosine distance)
        all_results.sort(key=lambda x: x[1])
        
        # Return top k documents
        top_docs = [doc for doc, score, ns in all_results[:self.k]]
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel search completed in {elapsed:.2f}s - found {len(all_results)} docs, returning top {len(top_docs)}")
        
        return top_docs

# Global chain cache for reuse
_chain_cache = {}

def setup_chain(site_name=None, force_new=False):
    """Setup chain with caching for faster subsequent calls."""
    global _chain_cache
    
    cache_key = site_name or "default"
    
    # Return cached chain if available and not forcing new
    if not force_new and cache_key in _chain_cache:
        logger.info(f"Returning cached chain for '{cache_key}'")
        return _chain_cache[cache_key]
    
    start_time = time.time()
    
    embeddings = get_embeddings()
    namespaces = get_all_namespaces()
    
    multi_retriever = FastMultiNamespaceRetriever(
        index_name=INDEX_NAME,
        embeddings=embeddings,
        namespaces=namespaces,
        k=RETRIEVAL_K
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    site_context = site_name if site_name else "the website"
    
    # OPTIMIZED: Shorter, more focused prompt = faster processing
    custom_prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=f"""You are a helpful support chatbot for Cool & Care (IT, security, HVAC solutions provider in DRC).

RULES:
- Be SPECIFIC with details (phone, email, services)
- Keep answers 2-4 sentences for simple questions, 4-6 for complex ones
- Use bullet points for lists (max 5 items)
- Include contact: +243 841 364 201, info@coolcare.cd when relevant

Context: {{context}}

Chat history: {{chat_history}}

Question: {{question}}

Answer (concise and specific):"""
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=multi_retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    # Cache the chain
    _chain_cache[cache_key] = chain
    
    elapsed = time.time() - start_time
    logger.info(f"Chain setup completed in {elapsed:.2f}s")
    
    return chain

def chat_step(chain, query, site_name=None, skip_hallucination_check=True):
    """Process chat query - optimized for speed."""
    start_time = time.time()
    logger.info(f"Processing query: {query}")
    
    # Get response from chain
    result = chain.invoke({"question": query})
    answer = result["answer"]
    sources = result["source_documents"]
    history = chain.memory.chat_memory.messages
    
    retrieval_time = time.time() - start_time
    logger.info(f"Chain response in {retrieval_time:.2f}s")
    
    # Log retrieved sources (minimal logging for speed)
    logger.info(f"Retrieved {len(sources)} sources")
    
    # SKIP hallucination check by default for speed
    # This saves ~1-2 seconds per query
    if ENABLE_HALLUCINATION_CHECK and not skip_hallucination_check and sources:
        is_grounded, explanation = check_hallucination_fast(answer, sources, query)
        if not is_grounded:
            answer = answer + "\n\n⚠️ Note: Please verify this information."
            logger.warning(f"Response may not be fully grounded")
    
    total_time = time.time() - start_time
    logger.info(f"Total response time: {total_time:.2f}s")
    
    return answer, history

def check_hallucination_fast(answer: str, sources: list, question: str):
    """Lightweight hallucination check - only if enabled."""
    # Use only first 2 sources and limited text for speed
    source_texts = "\n".join([doc.page_content[:300] for doc in sources[:2]])
    
    check_prompt = f"""Check if this answer is grounded in sources. Reply ONLY "Yes" or "No".

Answer: {answer[:500]}
Sources: {source_texts[:800]}

Grounded (Yes/No):"""
    
    try:
        # Use same LLM but with very low max_tokens
        response = get_llm().invoke(check_prompt)
        is_grounded = response.content.strip().lower().startswith("yes")
        return is_grounded, ""
    except Exception:
        return True, ""  # Fail-safe

# Async version for better performance with async frameworks
async def chat_step_async(chain, query, site_name=None):
    """Async version of chat_step for FastAPI."""
    loop = asyncio.get_event_loop()
    
    # Run the blocking chain.invoke in a thread pool
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: chain.invoke({"question": query})
        )
    
    answer = result["answer"]
    history = chain.memory.chat_memory.messages
    
    return answer, history

def clear_caches():
    """Clear all caches - useful for refreshing data."""
    global _chain_cache, _namespace_cache, _namespace_cache_time
    _chain_cache = {}
    _namespace_cache = {}
    _namespace_cache_time = 0
    logger.info("All caches cleared")

# Standalone testing
if __name__ == "__main__":
    print("Initializing chatbot (first time may take a few seconds)...")
    chain = setup_chain()
    print("Chatbot ready! (Type 'exit' to quit)")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
        if query.lower() == 'clear':
            clear_caches()
            chain = setup_chain(force_new=True)
            print("Caches cleared and chain reinitialized!")
            continue
            
        response, history = chat_step(chain, query)
        print("Bot:", response)
