import os
from dotenv import load_dotenv,find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

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
    if vectorstore is None:
        return False

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

def load_document(list_of_files):
    """Load documents from the specified directory."""
    
    # Read the text content from each file and store it with metadata
    documents = []
    for pdf_file in list_of_files:
        docs= load_single_pdf(pdf_file)
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    return docs


def load_or_create_faiss(filename,documents,store_name,db_dir,model_name):
    """Create a new FAISS store or update an existing one."""
    
    embeddings = get_embeddings(model_name)

    PERSIST_DIR = os.path.join(db_dir,store_name)
    
    vectorstore = load_faiss(store_name,db_dir,model_name)

    if vectorstore:
        print("Checking if file exists in FAISS DB...")

        if file_exists_in_db(vectorstore, filename):
            print(f" File '{filename}' already exists in vector DB. Skipping insert.")
            return vectorstore

        print("Adding new documents to existing DB...")
        vectorstore.add_documents(documents)
        vectorstore.save_local(PERSIST_DIR)
        return vectorstore

    else:
        print("Creating new FAISS DB...")
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


def rag_template():
    """Define the chat prompt template for the retrieval chain."""
    prompt =  ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful research assistant. Use the context provided to answer the question. "
            "If you don't know the answer, say you don't know. Be concise."
        ),
        MessagesPlaceholder(variable_name="context"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    return prompt


def create_huggingface_chain(model_name, db_dir, store_name):
    """Create a HuggingFace chat model and return a retrieval chain."""
    
    # Initialize HuggingFace chat model
    # llm
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        # task="conversational",
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        do_sample=False,
        provider="auto",  # let Hugging Face choose the best provider for you
    )   

    chat_model = ChatHuggingFace(llm=llm)
    
    # load vector store
    vectorstore = load_faiss(store_name, db_dir, model_name)
    if vectorstore is None:
        raise FileNotFoundError(f"Vector store '{store_name}' not found in '{db_dir}'")
 

    # load prompt template
    prompt = rag_template()

    # retriever
    retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})

    # Chain
    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | chat_model
    #     # | StrOutputParser()
    # )

    # prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a comedian who tells jokes about {topic}."),
    #         ("human", "Tell me {joke_count} jokes."),
    #     ]
    # )

    # # Create the combined chain using LangChain Expression Language (LCEL)
    # chain = prompt_template | chat_model | StrOutputParser()
    # chain = BM25Retriever.from_chain_type(
    #    llm=chat_model,
    #    chain_type="stuff",
    #    retriever=retriever,
    #    chain_type_kwargs={"prompt": prompt},
    # ) 

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain

def get_response(chain, query):
    """Get response from the chain for the given query."""
    # Run the chain
    # result = chain.run(query)
    result = chain.invoke({"input": query})
    # result = chain.invoke({"topic": query, "joke_count": 3})

    return result

if __name__ == "__main__":
    
    # app set up
    _ = load_dotenv(find_dotenv())
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    CURRENT_DIR = os.getcwd()
    # print(f"Current directory : {CURRENT_DIR}")
    DB_DIR = os.path.join(CURRENT_DIR, "db")

    STORE_NAME = "faiss_store"
    FILES_PATH = os.path.join(CURRENT_DIR,'docs')
    PERSIST_DIR = os.path.join(DB_DIR,STORE_NAME)

    # Choose a small HF model for quick embedding
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Create HuggingFace chain
    chain = create_huggingface_chain(EMBED_MODEL,DB_DIR,STORE_NAME)

    # Get response for a sample query
    query =  "What is the GPT model?"
    response = get_response(chain, query)
    print("Response:", response)