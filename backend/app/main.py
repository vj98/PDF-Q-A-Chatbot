import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset, concatenate_datasets
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader 
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
from statistics import mean
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking metrics
response_times = []
user_satisfaction_scores = []
correct_answers = 0
total_queries = 0

# Load the model and tokenizer
device = torch.device('cpu')
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float32)
persist_directory = "db"

# Fine-tuning function for two datasets (SQuAD and CNN/DailyMail)
def fine_tune_model():
    logger.info("Fine-tuning the model on two datasets...")

    # Step 1: Load SQuAD (for question answering) and CNN/DailyMail (for summarization)
    squad_dataset = load_dataset('squad')
    cnn_dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Step 2: Preprocess both datasets (tokenization)
    def preprocess_qa_function(examples):
        inputs = [q.strip() for q in examples['question']]
        targets = [a['text'][0].strip() for a in examples['answers']]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
        model_inputs['labels'] = labels
        return model_inputs

    def preprocess_summarization_function(examples):
        inputs = [article.strip() for article in examples['article']]
        targets = [summary.strip() for summary in examples['highlights']]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
        model_inputs['labels'] = labels
        return model_inputs

    # Apply preprocessing functions to both datasets
    tokenized_squad = squad_dataset.map(preprocess_qa_function, batched=True)
    tokenized_cnn = cnn_dataset.map(preprocess_summarization_function, batched=True)

    # Step 3: Combine both datasets for fine-tuning
    combined_train_dataset = concatenate_datasets([tokenized_squad["train"], tokenized_cnn["train"]])
    combined_eval_dataset = concatenate_datasets([tokenized_squad["validation"], tokenized_cnn["validation"]])

    # Step 4: Prepare data collator and training arguments
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True
    )

    # Step 5: Define Trainer with the combined dataset
    trainer = Seq2SeqTrainer(
        model=base_model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Step 6: Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    base_model.save_pretrained("fine-tuned-model")
    tokenizer.save_pretrained("fine-tuned-model")

    logger.info("Fine-tuning completed and model saved.")

# Data ingestion function
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
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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
    start_time = time.time()
    total_queries += 1
    try:
        qa = qa_llm()
        generated_text = qa(instruction)
        answer = generated_text['result']
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        logger.info(f"Answer generated successfully in {response_time:.2f} seconds.")
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
        return JSONResponse(status_code=200, content={"result": answer})
    except Exception as e:
        logger.error(f"Failed to retrieve answer: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def get_metrics():
    accuracy = (correct_answers / total_queries) * 100 if total_queries > 0 else 0
    avg_response_time = mean(response_times) if response_times else 0
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
    # Start fine-tuning before launching the API
    fine_tune_model()
    uvicorn.run("app:app", host="0.0.0.0", port=10000, log_level="info", reload=True)

if __name__ == "__main__":
    main()
