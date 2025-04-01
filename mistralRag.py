from dotenv import load_dotenv
import os
import streamlit as sl
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from mistralai import Mistral
import tempfile


def setUpInitialUI():
    # minimal UI
    sl.set_page_config(page_title="PDF AI")
    sl.header("Poses des question à ton PDF")
    
def main():
    
    setUpInitialUI()
       
    # load open ai api key for .env
    load_dotenv(os.getenv("OPENAI_API_KEY"))
    
    # load mistral
    mistral_api_key = os.getenv("MISTRAL_API_KEY")  
    mistral = Mistral(api_key=mistral_api_key)
    
    # upload the pdf file
    pdf = sl.file_uploader("Téléverse ton PDF", type="pdf")
    
    if pdf is not None:
         # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

            # Upload the file to Mistral OCR API
            with open(temp_file_path, "rb") as file_obj:
                
                uploaded_pdf = mistral.files.upload(
                    file={"file_name": pdf.name, "content": file_obj},
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

                #debug to see the ocr_response
                #sl.write(ocr_response)
                
                # extract content from pages
                pages = ocr_response.pages
                text_chunks = [page.markdown for page in pages]
                full_text = "\n\n".join(text_chunks)
                
                # create text splitter 
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                
                # make chunks using text splitter
                chunks = text_splitter.split_text(full_text)
                
                # create vector store
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
                
                #show user input 
                user_input = sl.text_input("Ask a question about your PDF:")
                
                #RAG Based on user input
                if user_input:
                    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
                    llm =ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini") # define llm and its parameters
                    
                    retriever = knowledge_base.as_retriever(search_kwargs={"k": 25}) # make it take 25 chucnks to provide more context
                    
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) # set up the conversation memory for the chat
      
                    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory) #build the chat system
   
                    response = chain.invoke({"question":user_input}) # send user prompt
                    
                    user_input=""
                     
                    sl.markdown(response["answer"])   #retrieve answer
                   
                

 
if __name__ == "__main__":
    main()  # Fixed: moved main() inside the if statement