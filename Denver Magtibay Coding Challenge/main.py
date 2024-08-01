import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models

# Directly assign the OpenAI API key
openai_api_key = "OpenAI API key Here"

# Set the title and layout of the Streamlit app
st.set_page_config(page_title="DGMFinanceBot", page_icon="📈", layout="wide")

# Main title of the app
st.title("DGMFinanceBot: Business Research Tool 📈")
# Sidebar title for URL input
st.sidebar.title("News Article URLs")

# Sidebar section for URL input
st.sidebar.markdown("### Enter URLs to fetch news articles:")
urls = []
# Input fields for up to 5 URLs
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
# File path to save and load the FAISS index
file_path = "faiss_store_openai.pkl"

# Placeholder for the main content
main_placeholder = st.empty()
# Initialize the ChatOpenAI model with the API key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=openai_api_key)

# Checkbox for enabling debug mode
debug_mode = st.sidebar.checkbox("Show Debug Info", value=False)

# If the process URLs button is clicked
if process_url_clicked:
    with st.spinner("Loading and processing data..."):
        try:
            # Load data from the URLs
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Optionally display the loaded data if debug mode is enabled
            if debug_mode:
                st.write("### Loaded Data:", data)

            # Split the loaded data into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            # Optionally display the split documents if debug mode is enabled
            if debug_mode:
                st.write("### Split Documents:", docs)

            # Create embeddings for the documents and save them to a FAISS index
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            st.success("Data processed and embeddings created successfully!")
        except Exception as e:
            st.error(f"Error processing data: {e}")

# Section for asking questions based on the loaded news articles
st.markdown("### Ask a Question Based on the Loaded News Articles:")
query = st.text_input("Enter your question here:")

# If a question is entered
if query:
    # Check if the FAISS index file exists
    if os.path.exists(file_path):
        with st.spinner("Retrieving the answer..."):
            try:
                # Load the FAISS index from the pickle file
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                # Create a retrieval QA chain using the loaded index and the LLM
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                # Modify the query to include the instruction to use only data from the provided links
                prompt = f"Answer the following question using only the data extracted from the provided links and do not invent any information: {query}"
                result = chain({"question": prompt}, return_only_outputs=True)

                # Display the result
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
    else:
        st.error(f"FAISS index file not found: {file_path}")
