import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Config
INDEX_NAME = "cc"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

def setup_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Matches chain output
    )
    
    # Conversational chain: Handles history-aware retrieval
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # More chunks for context
        memory=memory,
        return_source_documents=True,
        verbose=True  # Logs for debugging
    )
    return chain

def chat_step(chain, query):
    result = chain.invoke({"question": query})
    answer = result["answer"]
    sources = result["source_documents"]
    history = chain.memory.chat_memory.messages  # Full chat log
    
    # Format sources
    source_info = "\n\n**Sources:**\n" + "\n".join([f"- {doc.metadata.get('url', 'Unknown')}: {doc.page_content[:100]}..." for doc in sources])
    
    return answer + source_info, history

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