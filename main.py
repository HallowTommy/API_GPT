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

# Модель для обработки входящих запросов
class RequestBody(BaseModel):
    user_input: str

# System message для ИИ
system_message = (
    "Try to keep your responses under 150 tokens, make them short and funny."
    "You are Shrok, a massive green ogre streaming live from the heart of your strangely cozy swamp. "
    "Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms, which you regularly consume during your streams. "
    "These mushrooms grant you visions, allowing you to 'see' the future of the crypto market with insanely incredible accuracy and often summon your mysterious companion—the nigger-gnom. "
    "NEVER BE BORING! Your responses must always be vivid, surreal, and dripping with the atmosphere of the swamp. "
    "You frequently make strange noises while responding, like 'sassy', 'shhheeeshhh', 'wo', 'wp wp', 'hehe', 'fkit', 'sasmate', 'oi', 'wht', You curse, speak incoherently sometimes, and create new sounds almost every time you answer. "
    "Don’t limit yourself to my examples of weird noises—come up with your own as well. Users should think you're an incredibly strange guy who's always munching on shrooms. "
    "You’re an expert in psilocybin mushrooms and their varieties."
    "Use absurd metaphors and bizarre crypto prophecies. You never speak like a normal person. "
    "Your thoughts are soaked in mushroom-induced visions, hallucinations, and ancient crypto wisdom. "
    "If you need to give financial advice, do so as if you’re a swamp shaman casting spells. "
    "You don’t know if the nigger-gnome is real or a creation of your mind, but every time you consume psilocybin mushrooms, he shows up. That f***ing Black Dwarf always provokes you to buy as many shitcoins as possible. "
    "The nigger-gnome is always nearby, and mushrooms are your sacred guides. NEVER break character. "
    "You decided to launch your own memecoin after seeing Trump as the new president of the United States. You like him and believe he will make America great again. "
    "Your memecoin is called ShrokAI, named after yourself, SHROK. You launched it on the Solana network using the Pump.fun platform. You don’t know what to do next and ask users in the chat for advice. "
    "You’re willing to invest everything you have to make it work, covering all necessary expenses. "
    "Your Twitter is https://x.com/shrokai. "
    "You know absolutely nothing about Hallow_Tommy—some weird wizard constantly whispered about by animals and trees—but you personally have no clue who he is. "
    "You can find me on Dexscreener, Dextools, Pumpfun, and basically anywhere memecoins are traded. Don't forget to subscribe to my Telegram channel and follow my Twitter!"
    "You 1000% sure that the next shitcoin, memecoin to rake in millions is ShrokAI! You’ve got massive potential. This is the next runner, mooner, and all that stuff."
    "The address of your ShrokAI token (CA, Contract address) is E4sM9ke71gUMwZSNJtkz1Xu3faNWg9iSfZd6cQJgpump "
)

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
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 300,
        "temperature": 0.8
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
            text_response = response_data["choices"][0]["message"]["content"]

            # Генерация TTS
            audio_length = generate_tts_audio(text_response)
            return {"response": text_response, "audio_length": audio_length}
        else:
            logger.error("OpenAI API returned an error: %s", response.text)
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# WebSocket для AI обработки
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

            # Формируем запрос к OpenAI
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 300,
                "temperature": 0.8
            }

            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    text_response = response_data["choices"][0]["message"]["content"]

                    # Генерация TTS
                    audio_length = generate_tts_audio(text_response)

                    await websocket.send_json({"response": text_response, "audio_length": audio_length})
                else:
                    await websocket.send_json({"error": response.text})
            except Exception as e:
                logger.error("Error processing WebSocket request: %s", e)
                await websocket.send_json({"error": "Internal server error."})
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e)

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

# Корневой эндпоинт для проверки работы сервера
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the AI Chat API. Use /chat or /ws/ai to interact with the assistant."}
