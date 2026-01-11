import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import streamlit as st

def get_embeddings(model_name):
    """Return HuggingFace embeddings instance."""
    return HuggingFaceEmbeddings(model_name=model_name)

def load_faiss(store_name,db_dir,model_name):
    
    # embeddings
    embeddings = get_embeddings(model_name)
    
    PERSIST_DIR = os.path.join(db_dir,store_name)
    if os.path.exists(PERSIST_DIR):
        print("Existing FAISS DB found. Loading...")
        return FAISS.load_local(
            PERSIST_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def file_exists_in_db(vectorstore, filename):
    """Check metadata in DB to see if filename already exists."""
    print("Checking existing documents in vector store...")
    if vectorstore is None:
        return False
    
    filename = os.path.basename(filename)
    # metadata is stored inside docstore
    for _id, doc in vectorstore.docstore._dict.items():
        if doc.metadata.get("source") == filename:
            return True
    return False

def load_single_pdf(file_path):
    """Load a single PDF document."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    pdf_file = os.path.basename(file_path)
    documents = []
    # load docs
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        # Add metadata to each document indicating its source
        doc.metadata = {"source": pdf_file}
        documents.append(doc)
    return documents


def load_or_create_faiss(filename,documents,store_name,db_dir,model_name):
    """Create a new FAISS store or update an existing one."""
    
    embeddings = get_embeddings(model_name)

    PERSIST_DIR = os.path.join(db_dir,store_name)
    
    vectorstore = load_faiss(store_name,db_dir,model_name)

    if vectorstore:
        # Checking if file exists in FAISS DB
        if file_exists_in_db(vectorstore, filename):
            print(f"File '{filename}' already exists in vector DB. Skipping insert.")
            st.info(f"File '{filename}' already exists in vector DB. Skipping insert.")
            return vectorstore

        print("Adding new documents to existing DB...")
        vectorstore.add_documents(documents)
        vectorstore.save_local(PERSIST_DIR)
        return vectorstore

    else:
        print("Creating new FAISS DB...")
        st.info("Creating new FAISS DB...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(PERSIST_DIR)
        print(" Vector store is ready")
        return vectorstore

def db_pipeline(list_of_files,store_name,db_dir,model_name):
    """Load documents based on file type."""
    for pdf_file in list_of_files:

        documents= load_single_pdf(pdf_file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        documents = splitter.split_documents(documents)
        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(documents)}")

        vectorstore = load_or_create_faiss(pdf_file,documents,store_name,db_dir,model_name)
    return vectorstore


def rag_template_conversational():
    """Define the chat prompt template for the conversational retrieval chain."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful research assistant. Use the context provided to answer the question. "
            "If you don't know the answer, say you don't know. Be concise."
            "If the question is not related to the context, politely respond that you don't know"
            "\n\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"), # Add this line
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    return prompt

def create_huggingface_chain_conversational(model_name, db_dir, store_name):
    """Create a HuggingFace chat model and return a *conversational* retrieval chain."""
    
    # Initialize HuggingFace chat model (same as before)
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.01,
        repetition_penalty=1.03,
        do_sample=False,
        provider="auto",
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    # Load vector store and retriever (same as before)
    vectorstore = load_faiss(store_name, db_dir, model_name)
    if vectorstore is None:
        raise FileNotFoundError(f"Vector store '{store_name}' not found in '{db_dir}'")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create History-Aware Retriever Chain ---
    # This chain rephrases follow-up questions using chat history so retrieval is accurate
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Standalone question:")
    ])
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # Create main QA chain using the conversational prompt ---
    qa_prompt = rag_template_conversational()
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    
    # Combine the chains
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with RunnableWithMessageHistory to manage chat history
    store = {} 
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history", 
        output_key="output",
    )

    return conversational_rag_chain


def get_response_conversational(chain, query, session_id, history_store):
    """Get response from the *conversational* chain for the given query."""
    # Pass the session configuration when invoking the chain
    result = chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    
    # Store history
    history_store.append(HumanMessage(content=query))
    history_store.append(AIMessage(content=result['answer']))

    return result['answer']


if __name__ == "__main__":

    # app set up
    _ = load_dotenv(find_dotenv())
    CURRENT_DIR = os.getcwd()
    DB_DIR = os.path.join(CURRENT_DIR, "db")
    STORE_NAME = "faiss_store"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Create HuggingFace conversational chain
    chain = create_huggingface_chain_conversational(EMBED_MODEL,DB_DIR,STORE_NAME)
    
    # This dictionary simulates Streamlit's session state for tracking history
    mock_session_history = [] 
    
    # Example conversation:
    query1 = "What is the GPT model?"
    response1 = get_response_conversational(chain, query1, "user_abc", mock_session_history)
    print(f"User: {query1}")
    print(f"AI: {response1}\n")

    query2 = "How many parameter present in the GPT model?" # Follow-up question relying on previous context
    response2 = get_response_conversational(chain, query2, "user_abc", mock_session_history)
    print(f"User: {query2}")
    print(f"AI: {response2}\n")

    # Example of how mock_session_history looks
    # print(mock_session_history)
