from langchain.chains import ConversationalRetrievalChain
from all_prompt import general_prompt


def get_response(llm_model, vectorstore, query, chat_history):

    qa_chain_memory = ConversationalRetrievalChain.from_llm(
        llm_model,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": general_prompt()},
    )

    result = qa_chain_memory.invoke({'question': query, "chat_history": chat_history})

    return result["answer"]
