import streamlit as st
import time
from query import setup_chain, chat_step, clear_caches

# Page config
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Website AI Chatbot")

# ============= PERFORMANCE OPTIMIZATIONS =============
# 1. Chain initialization only once per session
# 2. Cached chain persists across reruns
# 3. Lazy loading of ingest module
# 4. Progress indicators for better UX

# Initialize chain once - cached in session state
@st.cache_resource(show_spinner=False)
def get_cached_chain():
    """Get cached chain - initialized only once."""
    return setup_chain()

# Setup chain with loading indicator (only on first load)
if "chain" not in st.session_state:
    with st.spinner("🚀 Initializing chatbot (first time only)..."):
        st.session_state.chain = get_cached_chain()
        st.session_state.chain_ready = True

# Sidebar for ingestion
with st.sidebar:
    st.header("⚙️ Settings")
    
    # URL ingestion (lazy loaded)
    st.subheader("📥 Ingest New URL")
    url = st.text_input("Website URL", placeholder="https://example.com")
    
    if st.button("🔄 Scrape & Store", disabled=not url):
        if url:
            with st.spinner("Scraping website... This may take a few minutes."):
                try:
                    # Lazy import to avoid loading on every run
                    from ingest import crawl_and_process
                    import asyncio
                    asyncio.run(crawl_and_process(url))
                    st.success("✅ Website scraped and stored!")
                    
                    # Clear caches to load new data
                    clear_caches()
                    st.cache_resource.clear()
                    st.session_state.pop("chain", None)
                    st.info("Please refresh the page to use new data.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "chain" in st.session_state:
            st.session_state.chain.memory.clear()
        st.success("Chat cleared!")
        st.rerun()
    
    # Refresh chain button
    if st.button("🔄 Refresh Chatbot"):
        with st.spinner("Refreshing..."):
            clear_caches()
            st.cache_resource.clear()
            st.session_state.pop("chain", None)
            st.session_state.chain = setup_chain(force_new=True)
        st.success("Chatbot refreshed!")
        st.rerun()
    
    st.divider()
    st.caption("💡 Tip: First response may be slower as the model warms up.")

# Chat interface with history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "response_time" in message:
            st.caption(f"⏱️ {message['response_time']}")

# Chat input
if prompt := st.chat_input("Ask about the website..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.spinner("Thinking..."):
            try:
                response, _ = chat_step(st.session_state.chain, prompt)
                response_time = time.time() - start_time
                response_time_str = f"{response_time:.1f}s"
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
                response_time_str = "Error"
        
        st.markdown(response)
        st.caption(f"⏱️ {response_time_str}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "response_time": response_time_str
        })

# Footer
st.divider()
st.caption("🔒 Your conversations are not stored permanently.")
