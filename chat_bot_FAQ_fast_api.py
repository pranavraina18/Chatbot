import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import asyncio

# Configure logging
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "chatbot_logs.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Global Variables
faq_data = None
vectorstore = None
executor = ThreadPoolExecutor(max_workers=10)  # Allows multi-threading

def get_questions_answers(url: str) -> dict:
    """ Fetch and parse FAQ data from the website """
    faq_data = {}
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    faq_blocks = soup.select('div.aos-init')
    for block in faq_blocks:
        question = block.find('summary')
        answer = block.find('p')
        if question and answer:
            faq_data[question.get_text(strip=True)] = answer.get_text(separator=" ", strip=True)

    return faq_data

def store_faq_in_vector_db(faq_data: dict):
    """ Store FAQs in a FAISS vector database """
    global vectorstore
    docs = [Document(page_content=q, metadata={"answer": a}) for q, a in faq_data.items()]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faq_vector_db")

def load_vectorstore():
    """ Load FAISS database into memory only if it exists """
    global vectorstore
    vector_db_path = "faq_vector_db"

    if not os.path.exists(vector_db_path) or not os.path.exists(f"{vector_db_path}/index.faiss"):
        logging.error("FAISS vector database not found! Creating a new one...")
        faq_data = get_questions_answers("https://aoltoronto.com/faq/")  # Re-fetch FAQ data
        store_faq_in_vector_db(faq_data)  # Store in FAISS again

    vectorstore = FAISS.load_local(vector_db_path, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
    
def retrieve_faq_answer(user_input: str) -> str:
    """ Retrieve the most relevant FAQ answer from FAISS """
    global vectorstore
    if vectorstore is None:
        load_vectorstore()

    results = vectorstore.similarity_search_with_score(user_input, k=3)
    if results:
        best_result = min(results, key=lambda x: x[1])  # Lower score = better match
        if best_result[1] < 0.3:
            return best_result[0].metadata["answer"]

    return None  # No matching FAQ found

def ask_llm(user_input: str) -> str:
    """ Generate a response using OpenAI's ChatGPT """
    prompt = ChatPromptTemplate([
        ("system", "You are a helpful FAQ assistant."),
        ("system", "If you know the answer, respond accordingly."),
        ("system", "If the question is unrelated to the FAQ topics, say: 'I'm sorry, I don't have information about that. Please contact us at info@aoltoronto.com or give us a call at (416) 969-8845.'"),
        ("user", "{question}")
    ])

    formatted_prompt = prompt.format(question=user_input)
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
    response = llm.invoke(formatted_prompt)

    return response.content

def hybrid_response(user_input: str) -> str:
    """ Use FAISS for fast retrieval; fallback to LLM if needed """
    faq_answer = retrieve_faq_answer(user_input)
    if faq_answer:
        return faq_answer

    llm_answer = ask_llm(user_input)
    if "I'm sorry" in llm_answer:
        log_interaction(user_input, "Out-of-scope question")

    return llm_answer

def log_interaction(user_input: str, response: str) -> None:
    """ Log interactions for debugging and improvement """
    logging.info(f"User: {user_input} | Bot: {response}")

@app.get("/ask")
async def welcome_message():
    """ API root message """
    return JSONResponse(content={"answer": "Welcome to the FAQ chatbot! How can I help you today?"})

@app.post("/ask")
async def ask(user_input: dict):
    """ Handle user questions asynchronously """
    try:
        question = user_input.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        # Check for exit keywords
        exit_keywords = ["exit", "bye", "quit", "goodbye", "end chat"]
        if question.strip().lower() in exit_keywords:
            return JSONResponse(content={"answer": "Thank you for chatting with me! Have a great day. 😊"})

        # Run in a thread pool for better concurrency
        response = await run_in_thread(hybrid_response, question.lower())
        log_interaction(question, response)
        return JSONResponse(content={"answer": response})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")

async def run_in_thread(func, *args):
    """ Run blocking functions in a thread """
    loop = asyncio.get_running_loop()  # ✅ Get the current event loop
    return await loop.run_in_executor(executor, func, *args)


if __name__ == "__main__":
    url = "" #dd url of fqa here

    logging.info("Fetching FAQ data...")
    faq_data = get_questions_answers(url)

    if faq_data:
        logging.info("Storing FAQ data in vector database...")
        store_faq_in_vector_db(faq_data)

    logging.info("Loading FAISS vector database into memory...")
    load_vectorstore()

    logging.info("FastAPI app is starting...")   
    uvicorn.run("chat_bot_FAQ_fast_api:app", host="127.0.0.1", port=5000)
    
    
    ### to start the code use --> uvicorn chabot.chat_bot_FAQ_fast_api:app --host 127.0.0.1 --port 5000 
    ## to ask a question using cli curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d "{\"question\": \"Can I Afford an Education in Canada?\"}"
