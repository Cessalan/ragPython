from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import tempfile

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from mistralai import Mistral

load_dotenv()  # Load API keys

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your React frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory store per session (simple version for demo)
memory_store = {}

class QAResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QAResponse)
async def ask_pdf(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...)  # Could come from frontend per user/browser
):
    # Step 1: Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        temp_file_path = tmp.name

    # Step 2: Mistral OCR
    mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    with open(temp_file_path, "rb") as f:
        uploaded_pdf = mistral.files.upload(
            file={"file_name": file.filename, "content": f},
            purpose="ocr"
        )
        signed_url = mistral.files.get_signed_url(file_id=uploaded_pdf.id)
        ocr_response = mistral.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=False,
        )

    # Step 3: Extract text from OCR response
    pages = ocr_response.pages
    full_text = "\n\n".join([page.markdown for page in pages])

    # Step 4: Chunk text
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)

    # Step 5: Embed and create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Step 6: Manage session memory
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

    memory = memory_store[session_id]

    # Step 7: Build RAG chain
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Step 8: Ask the question
    result = chain.invoke({"question": question})

    return QAResponse(answer=result["answer"])
