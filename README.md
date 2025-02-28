# FAQ Chatbot using FastAPI, FAISS, and OpenAI

## Overview
This project is a chatbot built using FastAPI that provides FAQ-based responses using FAISS for fast retrieval and OpenAI's ChatGPT for fallback answers. The bot scrapes FAQ data from a given website, stores it in a FAISS vector database, and serves user queries via a REST API.

## Features
- **Scrapes FAQ data** from a specified website.
- **Stores FAQ data** in a FAISS vector database for efficient retrieval.
- **Handles user queries** by first searching FAISS; if no relevant answer is found, it uses OpenAI's GPT-4.
- **FastAPI-powered API** with support for asynchronous processing.
- **ThreadPoolExecutor** for concurrent request handling.
- **Logs interactions** for debugging and improvement.

## Requirements
Ensure you have the following installed:

- Python 3.9+
- FastAPI
- Uvicorn
- BeautifulSoup4
- Requests
- FAISS
- LangChain
- OpenAI API Key
- dotenv

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/faq-chatbot.git
   cd faq-chatbot
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your OpenAI API key:
   ```ini
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the chatbot API:
   ```sh
   uvicorn chat_bot_FAQ_fast_api:app --host 127.0.0.1 --port 5000
   ```

2. Test the API using `curl`:
   ```sh
   curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "Can I afford an education in Canada?"}'
   ```

## API Endpoints

### `GET /ask`
Returns a welcome message.

### `POST /ask`
Handles user questions and returns the best-matching FAQ response.

#### Request Body:
```json
{
  "question": "Your question here"
}
```

#### Response:
```json
{
  "answer": "Relevant answer from FAQ or AI-generated response."
}
```

## Deployment
For production, you may want to use Gunicorn:
```sh
uvicorn chat_bot_FAQ_fast_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Logging
Logs are stored in `chatbot_logs.log` in the same directory as the script.

