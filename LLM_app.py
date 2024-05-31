import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY','')

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    st.session_state.final_document = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )

    st.session_state.vectorstore = FAISS.from_documents(
        st.session_state.final_document,
        st.session_state.embeddings
    )


st.title("ChatGroq Demo")
model_list = ["gemma-7b-it","mixtral-8x7b-32768"]

print("Selected Model--> ", model_list[1])

llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = model_list[1]
)
print("LLM: ", llm)

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on provided content only.
Please provide the most accurate response based on the question
<content>
{context}
</content>
Questions: {input}
"""
)


document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectorstore.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    print("Start time: ", start)
    response = retriever_chain.invoke({"input":prompt})
    print("Response time: ", time.process_time()-start)
    st.write(response['answer'])


    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------------------")


