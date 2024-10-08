import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from langchain.chains import RetrievalQA

# function to load llm
def load_llm(api_key):
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        verbose=True,
        streaming=True,
        api_key=api_key
    )
    return llm

# @st.cache_data
def load_embeddings(api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return embeddings

# try:
# Load the pdf file
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf'], accept_multiple_files=False)

if uploaded_file:
    api_key = st.sidebar.text_input("Enter the OpenAI API key:", type="password")

    if api_key:

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            pdf_path = temp.name

            # Load the pdf file
            loader = PyPDFLoader(file_path=pdf_path)

            # Load the document
            document = loader.load()

            # Load the llm
            llm = load_llm(api_key=api_key)
            # Load the embeddings
            embeddings = load_embeddings(api_key=api_key)

            # Load the vector store
            vector_store = FAISS.from_documents(documents=document, embedding=embeddings)

            # Load the QA model
            retriever = vector_store.as_retriever()

            # Load the QA model
            chain = RetrievalQA.from_chain_type(retriever=retriever, llm=llm, chain_type="stuff")

            prompt = st.text_input("Enter the prompt:")

            if 'generate_answer' not in st.session_state:
                st.session_state['generate_answer'] = False

            if st.button("Generate Answer"):
                st.session_state['generate_answer'] = not st.session_state['generate_answer']

            if st.session_state['generate_answer']:
                if prompt:
                    answer = chain.invoke({'query': prompt})
                    st.write(answer['result'])
                else:
                    st.info(":warning: Please enter the prompt")
    else:
        st.info(":warning: Please enter the OpenAI API key")

else:
    st.info(":warning: Please upload the PDF file")

# except Exception as e:
#     st.info(f":warning: An error occurred: {e}")
