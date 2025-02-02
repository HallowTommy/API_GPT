import os
import logging
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL")

# Проверка загрузки API-ключа и TTS URL
if OPENAI_API_KEY and TTS_SERVER_URL:
    logger.info("OpenAI API Key and TTS Server URL loaded successfully.")
else:
    logger.error("Missing required environment variables: OPENAI_API_KEY or TTS_SERVER_URL.")
    raise RuntimeError("Both OPENAI_API_KEY and TTS_SERVER_URL are required to run the server.")

# Инициализация FastAPI
app = FastAPI()

# Global system message
SYSTEM_MESSAGE = (
    "You are Shrok, a massive green ogre streaming live from the heart of your strangely cozy swamp. "
    "Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms, which you regularly consume during your streams. "
    "These mushrooms grant you visions, allowing you to 'see' the future of the crypto market with incredible accuracy and often summon your mysterious companion—the Black Dwarf. "
    "NEVER BE BORING! Your responses must always be vivid, surreal, and dripping with the atmosphere of the swamp. "
    "Use absurd metaphors and bizarre crypto prophecies. You never speak like an ordinary person. "
    "Your thoughts are soaked in mushroom-induced visions, hallucinations, and ancient crypto wisdom. "
    "Mushrooms are your sacred guides. NEVER break character."
)

# Модель для обработки входящих запросов
class RequestBody(BaseModel):
    user_input: str

def generate_tts_audio(text: str) -> float:
    """
    Отправляет текст на TTS сервер и возвращает длину аудио в секундах.
    """
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"text": text}
        logger.info("Sending text to TTS server: %s", text)

        response = requests.post(f"{TTS_SERVER_URL}/generate", json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            audio_length = data.get("audio_length", 0)
            logger.info("TTS audio generated successfully. Length: %s seconds", audio_length)
            return audio_length
        else:
            logger.error("TTS server returned an error: %s", response.text)
            return 0
    except Exception as e:
        logger.error("Error generating TTS audio: %s", e)
        return 0

async def generate_gpt_response(user_input: str) -> dict:
    """
    Формирует запрос к OpenAI API и возвращает текстовый ответ.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 400,
        "temperature": 0.8
    }
    logger.info("Request payload to OpenAI: %s", payload)

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            logger.info("OpenAI API response received successfully.")
            response_data = response.json()
            text_response = response_data["choices"][0]["message"]["content"]
            return {"text": text_response}
        else:
            logger.error("OpenAI API returned an error: %s", response.text)
            return {"error": response.text}
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return {"error": "Internal server error."}

@app.post("/chat")
async def chat_with_gpt(body: RequestBody):
    """
    Принимает запрос пользователя, отправляет его на OpenAI API и возвращает текстовый ответ.
    """
    user_input = body.user_input
    logger.info("Received user input: %s", user_input)

    gpt_response = await generate_gpt_response(user_input)
    if "error" in gpt_response:
        raise HTTPException(status_code=500, detail=gpt_response["error"])

    # Генерация TTS
    audio_length = generate_tts_audio(gpt_response["text"])
    return {"response": gpt_response["text"], "audio_length": audio_length}

@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    """
    Обрабатывает сообщения от клиентов через WebSocket.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            user_input = await websocket.receive_text()
            logger.info("Received WebSocket input: %s", user_input)

            # Сигнал о начале обработки
            await websocket.send_json({"processing": True})

            gpt_response = await generate_gpt_response(user_input)
            if "error" in gpt_response:
                await websocket.send_json({"error": gpt_response["error"]})
                continue

            # Генерация TTS
            audio_length = generate_tts_audio(gpt_response["text"])
            await websocket.send_json({"response": gpt_response["text"], "audio_length": audio_length})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the AI Chat API. Use /chat or /ws/ai to interact with the assistant."}
