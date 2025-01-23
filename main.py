import openai
import os
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL")  # URL TTS сервера

# Проверка API ключей
if not openai.api_key:
    logger.error("OpenAI API Key is missing.")
    raise RuntimeError("OpenAI API Key is required to run the server.")
if not TTS_SERVER_URL:
    logger.error("TTS Server URL is missing.")
    raise RuntimeError("TTS Server URL is required to run the server.")

logger.info("OpenAI API Key and TTS Server URL loaded successfully.")

# Инициализация FastAPI
app = FastAPI()

# Модель для обработки входящих запросов
class RequestBody(BaseModel):
    user_input: str

# Модель для ответа
class TTSResponse(BaseModel):
    response: str
    audio_length: float

# Функция для отправки текста в TTS и получения длины аудио
def send_to_tts(text: str) -> float:
    try:
        logger.info("Sending text to TTS: %s", text)
        response = requests.post(f"{TTS_SERVER_URL}/generate", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            logger.info("TTS response received: %s", data)
            return data.get("audio_length", 0.0)
        else:
            logger.error("TTS request failed: %s", response.text)
    except Exception as e:
        logger.error("Error sending to TTS: %s", str(e))
    return 0.0

# Эндпоинт для взаимодействия с OpenAI GPT и TTS
@app.post("/chat", response_model=TTSResponse)
async def chat_with_gpt(body: RequestBody):
    """
    Принимает запрос пользователя, отправляет его на OpenAI API и затем в TTS.
    Возвращает текстовый ответ и длину аудио.
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
        # Отправка запроса в OpenAI API
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",  # Или "gpt-4"
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        text_response = response["choices"][0]["message"]["content"]
        logger.info("OpenAI response: %s", text_response)

        # Отправка текста в TTS для генерации аудио
        audio_length = send_to_tts(text_response)

        return {"response": text_response, "audio_length": audio_length}

    except openai.error.OpenAIError as e:
        logger.error("OpenAI API error: %s", e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# WebSocket для взаимодействия
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получение сообщения от клиента
            message = await websocket.receive_text()
            logger.info("WebSocket received message: %s", message)

            # Генерация ответа GPT
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]

            try:
                gpt_response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                text_response = gpt_response["choices"][0]["message"]["content"]

                # Генерация аудио через TTS
                audio_length = send_to_tts(text_response)

                # Отправка ответа клиенту
                await websocket.send_json({
                    "response": text_response,
                    "audio_length": audio_length
                })

            except openai.error.OpenAIError as e:
                logger.error("OpenAI WebSocket error: %s", e)
                await websocket.send_json({"error": f"OpenAI error: {e}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")

# Корневой эндпоинт для проверки работы сервера
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the AI Chat API. Use /chat or /ws/chat to interact with the assistant."}
