import os
import base64
import time  # Import time module for tracking response times
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from constants import CHROMA_SETTINGS
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import logging
from typing import Optional
from statistics import mean  # For calculating averages
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking metrics
response_times = []  # To track query response times
user_satisfaction_scores = []  # To track user satisfaction ratings
correct_answers = 0  # To track correct responses for accuracy
total_queries = 0  # To track total number of queries processed

# Load the model and tokenizer
device = torch.device('cpu')
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float32)
persist_directory = "db"

def data_ingestion():
    logger.info("Starting data ingestion...")
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                logger.info(f"Ingesting file: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    # Create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None 
    logger.info("Data ingestion completed.")

def llm_pipeline():
    logger.info("Initializing LLM pipeline...")
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=1280,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("LLM pipeline initialized.")
    return local_llm

def qa_llm():
    logger.info("Initializing QA LLM...")
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    logger.info("QA LLM initialized.")
    return qa

def process_answer(instruction):
    global correct_answers, total_queries
    logger.info("Processing answer for the instruction...")
    start_time = time.time()  # Start tracking response time
    total_queries += 1
    try:
        qa = qa_llm()
        generated_text = qa(instruction)
        answer = generated_text['result']
        end_time = time.time()  # End tracking response time
        response_time = end_time - start_time
        response_times.append(response_time)  # Log response time
        logger.info(f"Answer generated successfully in {response_time:.2f} seconds.")
        # Simulate user feedback (this can be replaced with real feedback logic)
        # user_feedback = input("Was the answer correct? (yes/no): ").strip().lower()
        # if user_feedback == 'yes':
        #     correct_answers += 1
        return answer
    except Exception as e:
        logger.error(f"Error processing answer: {e}")
        raise e

@app.post("/upload/file")
def upload_file(uploaded_file: UploadFile = File(...)):
    logger.info("Received file upload request...")
    file_location = f"docs/{uploaded_file.filename}"
    path = "docs/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(uploaded_file.file, file_object)

    logger.info("File uploaded successfully. Starting data ingestion...")
    data_ingestion()

    return JSONResponse(status_code=200, content={"message": "File has been uploaded"})

@app.get("/query")
def query_response(query: Optional[str] = None):
    if query is None:
        return JSONResponse(status_code=500, content={"error": "No query provided"})

    try:
        answer = process_answer({'query': query})
        # Simulate user satisfaction feedback
        # satisfaction_score = int(input("Rate your satisfaction (1-5): "))
        # user_satisfaction_scores.append(satisfaction_score)
        return JSONResponse(status_code=200, content={"result": answer})
    except Exception as e:
        logger.error(f"Failed to retrieve answer: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def get_metrics():
    # Calculate accuracy
    accuracy = (correct_answers / total_queries) * 100 if total_queries > 0 else 0
    # Calculate average response time
    avg_response_time = mean(response_times) if response_times else 0
    # Calculate average user satisfaction
    avg_satisfaction = mean(user_satisfaction_scores) if user_satisfaction_scores else 0

    metrics = {
        "total_queries": total_queries,
        "correct_answers": correct_answers,
        "accuracy": f"{accuracy:.2f}%",
        "avg_response_time": f"{avg_response_time:.2f} seconds",
        "avg_user_satisfaction": f"{avg_satisfaction:.2f} / 5"
    }
    return JSONResponse(status_code=200, content=metrics)

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=10000, log_level="info", reload=True)

if __name__ == "__main__":
    main()
