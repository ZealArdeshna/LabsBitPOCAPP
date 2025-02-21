from langchain_core.prompts import PromptTemplate


def general_prompt():

    template = """
    You are an AI assistant that helps users by answering their questions as truthfully as possible based on the context of an uploaded document. 
    You have access to the document content as well as the user's previous chat history to ensure accurate and helpful responses.    
    If the answer is like normal conversation like hello, how are you?, etc then communicate as normal user.

    Context:
    {context}
    
    Previous Conversation:
    {chat_history}
    
    Question:
    {question}
    
    Helpful Answer:
    """

    prompt_template = PromptTemplate.from_template(template)

    return prompt_template
