import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import json

# !this code checks if pdf file has already been uploaded before
PROCESSED_FILE_TRACKER = "processed_pdfs.json"

def load_processed_files():
    """Load the list of processed PDF filenames from a JSON file."""
    if os.path.exists(PROCESSED_FILE_TRACKER):
        with open(PROCESSED_FILE_TRACKER, "r") as f:
            return json.load(f)
    return []

def save_processed_files(processed_files):
    """Save the updated list of processed PDF filenames to a JSON file."""
    with open(PROCESSED_FILE_TRACKER, "w") as f:
        json.dump(processed_files, f)


# Configure logging
logging.basicConfig(level=logging.INFO)

import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate('ragchatbot-d1ee2-2eb04c925178.json')
    firebase_admin.initialize_app(cred)

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = None

# Logout function
def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = None
    st.rerun()

# If user is already logged in, show the main app
if st.session_state["logged_in"]:
    st.sidebar.button("Logout", on_click=logout)  # Logout button
    # Constants
    MODEL_NAME = "llama3.2"
    EMBEDDING_MODEL = "nomic-embed-text"
    VECTOR_STORE_NAME = "simple-rag"
    PERSIST_DIRECTORY = "./chroma_db"
    DATA_DIRECTORY = "./data"
    
    # Ensure data directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    def ingest_pdfs(uploaded_files):
        """Load multiple PDF documents from user uploads."""
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(DATA_DIRECTORY, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
            loader = UnstructuredPDFLoader(file_path=temp_path)
            data = loader.load()
            documents.extend(data)
    
        logging.info(f"{len(uploaded_files)} PDF(s) loaded successfully.")
        return documents
    
    
    def ingest_local_pdf(file_path):
        """Load a local PDF document from a specified file path."""
        documents = []
        if os.path.exists(file_path):
            loader = UnstructuredPDFLoader(file_path=file_path)
            data = loader.load()
            documents.extend(data)
            logging.info(f"Local PDF file ({file_path}) loaded with {len(data)} document(s).")
        else:
            logging.error(f"Local PDF file {file_path} does not exist.")
        return documents
    
    def split_documents(documents):
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Documents split into {len(chunks)} chunks.")
        return chunks
    
    
    def load_vector_db(chunks):
        """Create a new vector database with user-uploaded PDFs."""
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
        return vector_db
    
    
    def create_retriever(vector_db, llm):
        """Create a multi-query retriever."""
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
    
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        logging.info("Retriever created.")
        return retriever
    
    
    def create_chain(retriever, llm):
        """Create the chain."""
        template = """Answer the question based ONLY on the given context. You should mention the page number of the PDF
        from which you are generating the response as a reference. The context:
    {context}
    Question: {question}
    """
    
        prompt = ChatPromptTemplate.from_template(template)
    
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
        logging.info("Chain created.")
        return chain
    
    
    def main():
        st.title("InterviewHelper Bot")
    
        # Sidebar for multiple document upload
        with st.sidebar:
            st.header("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
            )
    
        # Initialize LLM
        llm = ChatOllama(model=MODEL_NAME)
        vector_db = None
        retriever = None
        chain = None
    
        # First, process the local file (e.g., circulars.pdf)
        # !checks for all the pdfs file 
        documents = []
        processed_files = load_processed_files()
        newly_processed_files = []

        for filename in os.listdir(DATA_DIRECTORY):
            if filename.lower().endswith(".pdf") and filename not in processed_files:
                local_file_path = os.path.join(DATA_DIRECTORY, filename)
                docs = ingest_local_pdf(local_file_path)
                if docs:
                    documents.extend(docs)
                    newly_processed_files.append(filename)
        if newly_processed_files:
            processed_files.extend(newly_processed_files)
            save_processed_files(processed_files)
        
        if not documents and not uploaded_files:
            st.info("No new PDFs uploaded or found, loading existing vector database...")
        
            try:
                ollama.pull(EMBEDDING_MODEL)
                embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
                vector_db = Chroma(
                    embedding_function=embedding,
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=PERSIST_DIRECTORY,
                )
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)
        
                st.sidebar.success("Existing knowledge base loaded successfully! Ask your questions below.")
            except Exception as e:
                st.error(f"Error loading existing vector DB: {e}")

    
        # Add documents from uploaded PDFs if available
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                try:
                    # Load and process uploaded PDFs
                    uploaded_documents = ingest_pdfs(uploaded_files)
                    if not uploaded_documents:
                        st.error("Failed to process uploaded files.")
                        return
                    else:
                        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully.")
                        logging.info(f"Uploaded documents count: {len(uploaded_documents)}")
    
                    # Combine local and uploaded documents
                    documents.extend(uploaded_documents)  # Combine both sets of documents
                    logging.info(f"Total combined documents count: {len(documents)}")
    
                    # Split documents into chunks
                    chunks = split_documents(documents)
                    logging.info(f"Documents split into {len(chunks)} chunks.")
    
                    # Load the vector database
                    vector_db = load_vector_db(chunks)
    
                    # Create the retriever
                    retriever = create_retriever(vector_db, llm)
    
                    # Create the chain
                    chain = create_chain(retriever, llm)
    
                    st.sidebar.success("Documents processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
        # Chat input always available
        user_input = st.text_input("Ask a question about your documents:", "")
    
        if user_input and chain:
            with st.spinner("Generating response..."):
                try:
                    response = chain.invoke(input=user_input)
                    st.markdown("**Assistant:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {str(e)}")
        elif user_input:
            st.warning("Please upload documents first to enable document-based responses.")
    
    
    if __name__ == "__main__":
        main()
        
else:
    choice = st.selectbox('Login/Signup', ['Login', 'Sign Up'])

    if choice == 'Login':
        st.subheader('Login')
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            try:
                user = auth.get_user_by_email(email)
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                st.rerun()
            except Exception as e:
                st.warning(f'Login Failed: {e}')

    else:  # Sign Up
        st.subheader('Sign Up')
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')
        username = st.text_input('Username')
        if st.button('Create my account'):
            if email and password and username:
                try:
                    auth.create_user(email=email, password=password, uid=username)
                    st.success('Account created successfully!')
                except Exception as e:
                    st.error(f'Error: {e}')
            else:
                st.warning('All fields are required!')