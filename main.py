import openai
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Проверка загрузки API-ключа
if openai.api_key:
    logger.info("OpenAI API Key loaded successfully.")
else:
    logger.error("OpenAI API Key is missing. Check your .env file or environment variables.")
    raise RuntimeError("OpenAI API Key is required to run the server.")

# Инициализация FastAPI
app = FastAPI()

# Модель для входящих данных
class RequestBody(BaseModel):
    user_input: str

# Эндпоинт для обработки запросов к OpenAI GPT
@app.post("/chat")
async def chat_with_gpt(body: RequestBody):
    """
    Обрабатывает запрос пользователя, отправляет его на OpenAI API и возвращает ответ.
    """
    user_input = body.user_input
    logger.info("Received user input: %s", user_input)

    # Формирование данных для OpenAI API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    logger.info("Request payload to OpenAI: %s", messages)

    try:
        # Отправка запроса на OpenAI API
        response = await openai.ChatCompletion.acreate(  # Асинхронный вызов
            model="gpt-3.5-turbo",  # Или "gpt-4"
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        logger.info("OpenAI API response: %s", response)

        # Возвращение ответа пользователю
        return {"response": response["choices"][0]["message"]["content"]}

    except openai.OpenAIError as e:
        logger.error("OpenAI API error: %s", e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# Корневой эндпоинт для проверки работы сервера
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the AI Chat API. Use /chat to interact with the assistant."}
