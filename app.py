import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import faiss
from concurrent.futures import ThreadPoolExecutor

load_dotenv() # to see environment
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    def extract_text(pdf):
        pdf_reader = PdfReader(pdf)
        return "".join([page.extract_text() for page in pdf_reader.pages])

    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text, pdf_docs)
    
    text = "".join(results)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) # big pdf is there
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Extract vocabulary related to {field} in the {language} language from the provided context. Provide only the relevant words and their frequencies.\n\n
    Context:\n{context}\n
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "field", "language"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_vocabulary(text_chunks, field, language):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(f"Extract {field} vocabulary in {language}")
    chain = get_conversational_chain()
    context = "\n".join(text_chunks)
    response = chain({"input_documents": docs, "context": context, "field": field, "language": language}, return_only_outputs=True)
    return response["output_text"]

def format_output(vocabulary):
    lines = vocabulary.strip().split('\n')
    formatted = ""
    for line in lines:
        if ':' in line:
            word, frequency = line.split(':', 1)  # Split only on the first colon
            formatted += f"<li><strong>{word.strip()}:</strong> {frequency.strip()}</li>"
        else:
            formatted += f"<li><strong>{line.strip()}</strong></li>"  # Handle lines without a colon
    return f"<ul>{formatted}</ul>"

def main():
    st.set_page_config(page_title="Vocabulary Extractor", page_icon=":book:", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Extract Vocabulary from PDF 💁</h1>", unsafe_allow_html=True)

    language = st.text_input("Enter the language")
    field = st.text_input("Enter the specific field of vocabulary (e.g., health)")

    if 'vector_store_created' not in st.session_state:
        st.session_state.vector_store_created = False

    with st.sidebar:
        st.markdown("<h2 style='color: #FF5733;'>Menu</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.vector_store_created = True
                st.session_state.text_chunks = text_chunks
                st.success("Processing completed!")

    if language and field and st.session_state.vector_store_created:
        vocabulary = extract_vocabulary(st.session_state.text_chunks, field, language)
        formatted_vocabulary = format_output(vocabulary)
        st.markdown(f"<h3>Vocabulary related to <i>{field}</i> in <i>{language}</i>:</h3>", unsafe_allow_html=True)
        st.markdown(formatted_vocabulary, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
