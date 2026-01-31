
# ===========================================================================
# STREAMLIT RAG CHATBOT APP
# ===========================================================================
# A simple web interface for our customer support chatbot
# Run with: streamlit run app.py
# ===========================================================================

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Everstorm Outfitters Support")
st.caption("Ask me about shipping, returns, payments, and more!")

# ---------------------------------------------------------------------------
# LOAD RAG COMPONENTS (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_chain():
    """Load and cache the RAG chain components."""
    # Load embeddings model
    embedder = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

    # Load saved FAISS index
    vectordb = FAISS.load_local(
        "faiss_index", 
        embedder,
        allow_dangerous_deserialization=True  # Required for loading pickle files
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    # Initialize LLM
    llm = Ollama(model="gemma3:1b", temperature=0.1)

    # System prompt
    SYSTEM_TEMPLATE = """
    You are a helpful Customer Support Chatbot for Everstorm Outfitters.

    Rules:
    1. Use ONLY the provided context to answer.
    2. If unsure, say "I don't know based on the documents."
    3. Be concise and helpful.

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Build chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

# Load the chain
chain = load_chain()

# ---------------------------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------------------------

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about our policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            response = result["answer"]
            st.markdown(response)

    # Update history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append((prompt, response))

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions using RAG.")
    st.write("**Powered by:**")
    st.write("- 🦜 LangChain")
    st.write("- 📊 FAISS")
    st.write("- 🤖 Gemma 3 (via Ollama)")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
