## Importing libraries
import streamlit as st
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import  ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder  
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader

## Using Huggingface sentence transformer as embedding model 
os.environ['HF_TOKEN'] = 'Put your HuggingFace Token'
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

## Desgining streamlit framework
st.title('RAG Conversational QnA chatbot with History')
st.write('Upload PDFs and chat with your documents')

## Button to input your Groq API to call the LLM's
api_key = st.text_input('Enter your Groq API Key:', type='password')

if api_key:
    llm = ChatGroq(api_key= api_key,model='Gemma2-9b-It')

    #Chat interface
    session_id = st.text_input("Session ID", value="default session")
    
    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks,embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
        ("system",contextualize_q_system_prompt),
        (MessagesPlaceholder("chat_history")),
        ("human","{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever,contextualize_q_prompt)
    
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    if 'store' not in st.session_state:
        st.session_state.store = {}

    def get_session_history(session_id : str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key='input',
        output_messages_key= 'answer',
        history_messages_key='chat_history'
        )
    
    user_input = st.text_input('Your Question........')

    config = {"configurable": {"session_id":session_id}}

    if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.write('Please enter your Groq API Key')
    st.stop()

















