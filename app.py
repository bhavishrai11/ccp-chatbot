import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure the Streamlit page layout
st.set_page_config(page_title="CCP RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Live CCP RAG Chatbot")
st.markdown("Upload your CCP project documents and chat with them in real-time.")

# Sidebar setup for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    uploaded_file = st.file_uploader("Upload CCP Documents (PDF)", type="pdf")
    
    st.markdown("---")
    if st.button("🔄 Reset Chat & Vector Store"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.success("Cleared everything successfully!")
        st.rerun()

# Stop application execution if the API key is missing
if not gemini_api_key:
    st.info("⚠️ Please enter your Gemini API key in the sidebar to begin.", icon="🔑")
    st.stop()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process the uploaded file into the RAG system
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Processing and indexing your CCP data... Please wait."):
        # Temporarily save file to disk to pass to PyPDFLoader
        temp_file = "temp_uploaded.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # 1. Load the PDF document
            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            
            # 2. Chunk the text logically
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
     # 3. Generate embeddings and store them in local FAISS
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=gemini_api_key
            )
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.success("✅ File successfully indexed! Ready to query.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Render past chat logs
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user messages
if user_query := st.chat_input("Ask something about your CCP data..."):
    # Display human message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    google_api_key=gemini_api_key,
                    temperature=0.3
                )
                
                if st.session_state.vector_store is not None:
                    # RAG Mode: Search vector store and synthesize response
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    system_prompt = (
                        "You are an intelligent AI system specialized in answering questions about this CCP project document. "
                        "Rely strictly on the retrieved context to answer the question accurately. "
                        "If you don't know the answer or if it's not present in the context, say that you cannot find it in the provided documents.\n\n"
                        "Context:\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, document_chain)
                    
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                else:
                    # Fallback General Chat Mode if no document has been provided yet
                    response = llm.invoke(user_query)
                    answer = response.content
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
