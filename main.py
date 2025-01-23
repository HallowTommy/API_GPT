import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверка загрузки API-ключа
if OPENAI_API_KEY:
    logger.info("OpenAI API Key loaded successfully.")
else:
    logger.error("OpenAI API Key is missing.")
    raise RuntimeError("OpenAI API Key is required to run the server.")

# Инициализация FastAPI
app = FastAPI()

# Модель для обработки входящих запросов
class RequestBody(BaseModel):
    user_input: str

# Эндпоинт для взаимодействия с OpenAI GPT
@app.post("/chat")
async def chat_with_gpt(body: RequestBody):
    """
    Принимает запрос пользователя, отправляет его на OpenAI API и возвращает текстовый ответ.
    """
    user_input = body.user_input
    logger.info("Received user input: %s", user_input)

    # Формирование данных для OpenAI API
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",  # Или "gpt-4"
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    logger.info("Request payload to OpenAI: %s", payload)

    try:
        # Отправка запроса в OpenAI API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Проверка успешности запроса
        if response.status_code == 200:
            logger.info("OpenAI API response received successfully.")
            response_data = response.json()
            logger.info("Response data: %s", response_data)
            return {"response": response_data["choices"][0]["message"]["content"]}
        else:
            logger.error("OpenAI API returned an error: %s", response.text)
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# Корневой эндпоинт для проверки работы сервера
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the AI Chat API. Use /chat to interact with the assistant."}
