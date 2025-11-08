import streamlit as st
from query import setup_chain, chat_step  # Import from above

st.title("Website AI Chatbot")

# Setup chain once (shared across session)
if "chain" not in st.session_state:
    st.session_state.chain = setup_chain()

# Sidebar for ingestion (from previous)
with st.sidebar:
    from ingest import crawl_and_process  # Import async main as sync for simplicity
    url = st.text_input("Ingest new URL")
    if st.button("Scrape & Store") and url:
        import asyncio
        asyncio.run(crawl_and_process(url))  # Run async ingest
        st.success("Stored! Refresh chat for new data.")

# Chat interface with history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the website..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, _ = chat_step(st.session_state.chain, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.chain.memory.clear()  # Reset memory
    st.rerun()