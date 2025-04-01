from dotenv import load_dotenv
import os
import streamlit as sl
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

def main():
    # load open ai api key for .env
    load_dotenv(os.getenv("OPENAI_API_KEY"))
 
    # minimal UI
    sl.set_page_config(page_title="Ask PDF")
    sl.header("ask pdf")
    
    # upload the pdf file
    pdf = sl.file_uploader("Upload pdf", type="pdf")
    text = ""
    
    # extract the text from the pdf using PdfReader
    if pdf is not None:
        pdf_reader = PdfReader(pdf)     
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    #debug to check if text has been extracted
    #sl.write(text)
    
  
    if text != "":
        # split text into chunks using langchain  
        text_splitter = CharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap =200,
        )
        
        chunks = text_splitter.split_text(text)
    
        #debug to check if chunks are OK
        #sl.write(chunks)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        # create vector store by putting chunks through the embedding
        knowledge_base = FAISS.from_texts(chunks,embeddings)
        #knowledge_base = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")
        
        #show user input
        user_input = sl.text_input("Ask a question about your PDF:")
        
        if user_input:
            docs = knowledge_base.similarity_search(user_input)
            #sl.write(docs)
            
            llm =ChatOpenAI(temperature=0.5)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents =docs, question= user_input);
            
            sl.write(response)
    
main()