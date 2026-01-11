import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv

import helper 

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

# streamlit app
st.header('Research Assistant')
tab1, tab2, tab3 = st.tabs(['Chat','Upload Document','Chart'])

# Chat tab
with tab1:
    # ensure history exists
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of {"role": "user"|"assistant", "text": "..."}
    
    # show previous messages
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Assistant:** {msg['text']}")

    # Input box
    input_text = st.text_area("Input:", key='input')
    submit = st.button("Ask question")

    if submit and input_text:
        # append user message to history
        st.session_state["history"].append({"role": "user", "text": input_text})
        mock_session_history = [] 

        # create retrieval chain
        chain = helper.create_huggingface_chain_conversational(EMBED_MODEL, DB_DIR, STORE_NAME)
        # get response
        # response = helper.get_response(chain, input_text)
        response = helper.get_response_conversational(chain, input_text, "user_abc", mock_session_history)

        # append assistant response to history and display
        st.session_state["history"].append({"role": "assistant", "text": response})
        st.subheader("The Response is")
        st.write(response)

# Upload Document tab
with tab2:
    st.subheader("Upload Document")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Upload"):
        for uploaded_file in uploaded_files:
            with open(os.path.join(FILES_PATH, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

    # Load documents and create/load vector store
    if st.button("Create/Load Vector Store"):
        with st.spinner("Loading documents..."):
            # Get the list of uploaded file paths
            uploaded_files_path  = [os.path.join(FILES_PATH, w.name) for w in uploaded_files]
            
            # vector store creation/loading
            vector_store = helper.db_pipeline(uploaded_files_path,STORE_NAME,DB_DIR,EMBED_MODEL)
                
        st.success("Vector store is ready.")

# Chart tab
with tab3:
    st.subheader("Charts will be displayed here")