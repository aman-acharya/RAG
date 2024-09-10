import streamlit as st
import tempfile
from langchain_community.document_loaders import  PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

#function to load llm
def load_llm():
    llm = ChatOpenAI(
        model = "gpt-4-turbo",
        temperature=0,
        verbose=True,
        streaming = True,
        api_key= api
    )
    return llm

# @st.cache_data
def load_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings

# Load the pdf file
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf'], accept_multiple_files=False)

api = st.sidebar.text_input("Enter the openai api key : ", type="password")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.getvalue())
        pdf_path = temp.name

        # Load the pdf file
        loader = PyPDFLoader(
            file_path=pdf_path
        )

        # Load the document
        document = loader.load()

        # Load the llm
        llm = load_llm()

        # Load the embeddings
        embeddings = load_embeddings()

        persist_directory = tempfile.mkdtemp()

        if 'persist_directory' not in st.session_state:
            st.session_state['persist_directory'] = persist_directory
            
        # Load the vector store
        vector_store = Chroma.from_documents(
            documents=document,
            embedding=embeddings,
            persist_directory= persist_directory
        )

        # Load the QA model
        retriver = vector_store.as_retriever()

        # Load the QA model
        chain = RetrievalQA.from_chain_type(
            retriever= retriver,
            llm = llm,
            chain_type= "stuff",
        )

        prompt = st.text_input("Enter the prompt : ")

        if 'generate_answer' not in st.session_state:
            st.session_state['generate_answer'] = False
        
        if st.button("Generate Answer"):
            st.session_state['generate_answer'] = not st.session_state['generate_answer']

        if st.session_state['generate_answer']:
            if prompt is not None:
                answer = chain.invoke({'query': prompt})
                st.write(answer['result'])
            else:
                st.info("Please enter the prompt")
else:
    st.info("Please upload the pdf file")