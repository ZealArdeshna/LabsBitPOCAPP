import retrieval_qa
import streamlit as st
from llm_model import LLMModel
from embed_and_store import VectorStore

vector_store = VectorStore()
llm_models = LLMModel()

error_message = "Sorry, Something Went Wrong!"

st.header("LabsBit Test Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question related to document:"):

    try:
        vector_db = vector_store.load_faiss_db(vector_store.azure_openai_embedding())
        st.session_state.vector_db = vector_db
    except FileNotFoundError:
        st.error("Please upload a document first and click 'Submit' to create embeddings.")

    if st.session_state.vector_db:

        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                full_response = retrieval_qa.get_response(llm_models.azure_llm_model(),
                                                          st.session_state.vector_db,
                                                          user_question,
                                                          st.session_state.chat_history)
            except Exception as exe:
                print(exe)
                full_response = error_message

            message_placeholder.markdown(full_response)

        st.session_state.chat_history.extend([(user_question, full_response)])

        if len(st.session_state.chat_history) >= 10:
            st.session_state.chat_history.pop(0)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
