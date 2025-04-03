from dotenv import load_dotenv
import os
import streamlit as sl
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from mistralai import Mistral
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
import tempfile
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time


# Set up environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# We are gonna load multiple file types
def get_loader_for_file(file_path):
    """Return the appropriate loader based on file extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == '.pdf':
        return PyPDFLoader(file_path)
    elif extension == '.txt':
        return TextLoader(file_path)
    elif extension == '.csv':
        return CSVLoader(file_path)
    elif extension in ['.doc', '.docx']:
        return Docx2txtLoader(file_path)
    elif extension in ['.xls', '.xlsx']:
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    
# We are gonna process multiple file types
def process_file(file , file_type):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
    
    try:
        if file_type == "pdf":
            # Use Mistral OCR for PDF files
            mistral = Mistral(api_key=MISTRAL_API_KEY)
            
            # Upload the file to Mistral OCR API
            with open(temp_file_path, "rb") as file_obj:
                uploaded_pdf = mistral.files.upload(
                    file={"file_name": file.name, "content": file_obj},
                    purpose="ocr"
                )
                
                # Get signed URL for OCR processing
                signed_url = mistral.files.get_signed_url(file_id=uploaded_pdf.id)
                
                # Perform OCR using the signed URL
                ocr_response = mistral.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": signed_url.url,
                    },
                    include_image_base64=False,
                )

                # Extract content from pages
                pages = ocr_response.pages
                text_chunks = [page.markdown for page in pages]
                full_text = "\n\n".join(text_chunks)
                
                return full_text, file.name
        else:
            # For non-PDF files, use regular loaders
            dynamic_loader = get_loader_for_file(temp_file_path)
            documents = dynamic_loader.load()
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            return full_text, file.name  
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, file.name
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_file_path): #check if file exist
            os.unlink(temp_file_path) # delete it
        

            
        
    
