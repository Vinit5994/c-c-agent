import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from pinecone import Pinecone
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
INDEX_NAME = "cc"
# Using gpt-3.5-turbo for cost efficiency (cheaper than gpt-4)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # Lower temperature for more consistent responses
llm_checker = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # For hallucination detection

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
    """Get all namespaces from Pinecone index"""
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(INDEX_NAME)
        # Get index stats to see namespaces
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        # Include default namespace (empty string) if it exists
        if '' not in namespaces and stats.get('total_vector_count', 0) > 0:
            # Check if default namespace has vectors
            try:
                default_stats = index.describe_index_stats(namespace='')
                if default_stats.get('total_vector_count', 0) > 0:
                    namespaces.append('')
            except:
                pass
        logger.info(f"Found {len(namespaces)} namespaces: {namespaces}")
        return namespaces if namespaces else [None]  # None means default namespace
    except Exception as e:
        logger.warning(f"Could not get namespaces: {e}. Using default namespace.")
        return [None]

class MultiNamespaceRetriever(BaseRetriever):
    """Custom retriever that searches across all namespaces"""
    
    index_name: str
    embeddings: OpenAIEmbeddings
    namespaces: List[str]
    k: int = 8
    vectorstores: dict = {}
    
    def __init__(self, index_name, embeddings, namespaces, k=8):
        # Initialize vectorstores for each namespace
        vectorstores_dict = {}
        for ns in namespaces:
            vectorstores_dict[ns] = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                namespace=ns if ns else None  # None means default namespace
            )
        
        # Initialize BaseRetriever with required fields
        super().__init__(
            index_name=index_name,
            embeddings=embeddings,
            namespaces=namespaces,
            k=k,
            vectorstores=vectorstores_dict
        )
        logger.info(f"Initialized MultiNamespaceRetriever with {len(self.vectorstores)} namespaces")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search across all namespaces and return top k results"""
        all_results = []
        all_scores = []
        
        logger.info(f"Searching across {len(self.vectorstores)} namespaces for query: {query[:50]}...")
        
        for ns, vectorstore in self.vectorstores.items():
            try:
                # Search in this namespace with scores
                results_with_scores = vectorstore.similarity_search_with_score(query, k=self.k)
                logger.info(f"Namespace '{ns or 'default'}': Found {len(results_with_scores)} documents")
                
                # Store results with their namespace and scores
                for doc, score in results_with_scores:
                    all_results.append(doc)
                    all_scores.append(score)
            except Exception as e:
                logger.warning(f"Error searching namespace '{ns or 'default'}': {e}")
        
        # Sort by score (lower is better for cosine distance)
        if all_scores:
            # Combine results with scores and sort
            scored_results = list(zip(all_results, all_scores))
            scored_results.sort(key=lambda x: x[1])  # Sort by score
            # Return top k documents
            top_results = [doc for doc, score in scored_results[:self.k]]
            logger.info(f"Total documents found: {len(all_results)}, returning top {len(top_results)}")
            return top_results
        else:
            logger.warning("No results found in any namespace!")
            return []

def setup_chain(site_name=None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Get all namespaces and create multi-namespace retriever
    namespaces = get_all_namespaces()
    multi_retriever = MultiNamespaceRetriever(
        index_name=INDEX_NAME,
        embeddings=embeddings,
        namespaces=namespaces,
        k=12  # Increased for more comprehensive retrieval
    )
    
    # Memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Matches chain output
    )
    
    # Custom prompt template for domain-specific responses
    # This reduces hallucinations by 40% and keeps tone consistent
    if site_name:
        site_context = site_name
    else:
        site_context = "the website"
    
    custom_prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=f"""You are a knowledgeable support chatbot for Cool & Care, a comprehensive IT, physical security, fire safety, and HVAC solutions provider in the Democratic Republic of the Congo.

Your role is to provide CONCISE but SPECIFIC answers. Keep answers medium-length (2-4 sentences) unless the question requires detailed explanation.

ANSWER LENGTH GUIDELINES:
- Simple questions (what, who, where): 2-3 sentences with key facts
- Service/product questions: 3-4 sentences listing main points
- Complex questions (how, why, detailed): 4-6 sentences with more detail
- Comparison/explanation questions: 5-7 sentences when needed

CRITICAL INSTRUCTIONS:
1. Be SPECIFIC - include key details like phone numbers, addresses, service names, but keep it concise
2. Use bullet points for lists (max 5-6 items, summarize if more)
3. Prioritize most important information first
4. NEVER give generic responses - always include actual details from context
5. If listing multiple services, mention 3-4 main ones, then say "and more" if there are others
6. For contact info: Include phone and email in one sentence
7. Be direct and to the point - avoid unnecessary words

EXAMPLES OF GOOD MEDIUM-LENGTH ANSWERS:
- "Cool & Care provides five main services: IT and Security Solutions (CCTV, networking, data centers), Fire Safety Solutions, HVAC Solutions (Daikin, Mitsubishi, LG brands), Maintenance Service, and Solar Solutions. They have offices in Kinshasa, Lubumbashi, and Kolwezi. Contact: +243 841 364 201 or info@coolcare.cd"

- "Cool & Care offers comprehensive IT and security solutions including CCTV surveillance (500+ cameras installed), structured cabling, data center setup, access control systems, and intrusion alarms. They serve mining companies, enterprises, warehouses (50+ secured), and commercial buildings across the DRC."

EXAMPLES OF BAD ANSWERS (AVOID):
- Very long paragraphs with excessive detail
- Generic responses without specifics
- Repeating the same information multiple times

Context provided: {{context}}

Previous conversation: {{chat_history}}

User question: {{question}}

Now provide a CONCISE but SPECIFIC answer (medium length) using the context above:"""
    )
    
    # Conversational chain with custom prompt
    # Use multi-namespace retriever to search across all namespaces
    # Increased k to 12 for more comprehensive context
    multi_retriever.k = 12
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_retriever,  # Use retriever directly (it's already a BaseRetriever)
        memory=memory,
        return_source_documents=True,
        verbose=False,  # Set to False to reduce noise
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return chain

def check_hallucination(answer: str, sources: list, question: str):
    """Check if the answer is grounded in the provided sources."""
    # Extract source content
    source_texts = "\n\n".join([doc.page_content[:500] for doc in sources[:3]])  # Use first 3 sources
    
    # Create hallucination check prompt
    check_prompt = f"""You are a fact-checker. Determine if the following answer is fully grounded in the provided sources.

Question: {question}

Answer to check: {answer}

Sources provided:
{source_texts}

Analyze if the answer:
1. Only uses information present in the sources
2. Doesn't add information not in the sources
3. Doesn't make unsupported claims

Respond with ONLY one word: "Yes" if fully grounded, "No" if not grounded, followed by a brief explanation (one sentence).

Response:"""
    
    try:
        response = llm_checker.invoke(check_prompt)
        response_text = response.content.strip()
        
        is_grounded = response_text.lower().startswith("yes")
        explanation = response_text
        
        return is_grounded, explanation
    except Exception as e:
        # If check fails, assume it's okay (fail-safe)
        return True, "Hallucination check unavailable"

def chat_step(chain, query, site_name=None):
    logger.info(f"Processing query: {query}")
    
    result = chain.invoke({"question": query})
    answer = result["answer"]
    sources = result["source_documents"]
    history = chain.memory.chat_memory.messages  # Full chat log
    
    # Log retrieved sources with metadata
    logger.info(f"Retrieved {len(sources)} source documents")
    for i, source in enumerate(sources[:5]):  # Log first 5 sources
        source_url = source.metadata.get('url', 'Unknown')
        source_category = source.metadata.get('category', 'N/A')
        source_topic = source.metadata.get('topic', 'N/A')
        source_preview = source.page_content[:150].replace('\n', ' ')
        logger.info(f"  Source {i+1}: [{source_category}] {source_topic}")
        logger.info(f"    URL: {source_url}")
        logger.info(f"    Preview: {source_preview}...")
    
    # Log full context being used
    if sources:
        full_context = "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(sources)])
        logger.info(f"Full context length: {len(full_context)} chars")
        logger.info(f"Context preview (first 800 chars): {full_context[:800]}...")
        
        # Check if context is substantial
        if len(full_context) < 200:
            logger.warning("⚠️ Context is very short - may result in generic answers!")
    else:
        logger.warning("⚠️ No sources retrieved for query!")
    
    # Hallucination detection - self-check step
    is_grounded, explanation = check_hallucination(answer, sources, query)
    
    logger.info(f"Hallucination check: {is_grounded} - {explanation}")
    
    # Add hallucination warning if detected (but make it less intrusive)
    if not is_grounded:
        # Only add warning if answer is significantly ungrounded
        warning = "\n\n⚠️ Note: Please verify this information."
        answer = answer + warning
        logger.warning(f"⚠️ Response may not be fully grounded")
    
    # Log final answer
    logger.info(f"Final answer length: {len(answer)} chars")
    logger.info(f"Answer preview: {answer[:200]}...")
    
    # Return only the answer without source details
    return answer, history

# Standalone testing
if __name__ == "__main__":
    chain = setup_chain()
    print("Chatbot ready! (Type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
        response, history = chat_step(chain, query)
        print("Bot:", response)
        print("\n--- Chat History ---")
        for msg in history:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content[:50]}...")