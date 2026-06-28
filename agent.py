import logging
import os
import time
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.md"
KNOWLEDGE_PATH = BASE_DIR / "knowledge.md"
SCRIPTS_PATH = BASE_DIR / "scripts.md"
OFFERS_PATH = BASE_DIR / "offers.md"

AVITO_CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
AVITO_CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

FALLBACK_SYSTEM_PROMPT = """Ты ассистент магазина BXOKKEE / ВХОККЕЕ.
Отвечай коротко, спокойно и по делу. Не выдумывай цены, наличие, сроки, ссылки и совместимость.
Сначала закрывай основной вопрос клиента. Задавай максимум один уточняющий вопрос.
Если данных недостаточно, используй: Пришлите фото/модель — подскажу точнее.
"""

FALLBACK_KNOWLEDGE = """Адрес магазина: Новосибирск, ул. Богдана Хмельницкого, 27.
Профилирование: 1290 ₽.
Заточка: классическая — 400 ₽; новая пара — 590 ₽; фигурные — 490 ₽; ржавые / повреждённые — от 490 ₽.
Бренды лезвий писать строго: in hockey® Base и in hockey® Pro.
При профилировании заточка входит в подарок.
"""

FALLBACK_SCRIPTS = """Пришлите фото/модель — подскажу точнее.
Можем сразу подготовить лезвия к выходу на лёд: сделать профилирование и заточку нужным желобом. При профилировании заточка входит в подарок — получите лезвия уже готовыми к катанию.
"""

FALLBACK_OFFERS = """# Текущие акции и специальные предложения

## Активные предложения

Пока активных специальных предложений нет.

## Правила

Все цены из knowledge.md считать базовыми. Если активной акции нет, не выдумывать скидку. UDS — отдельная бонусная система.
"""

FORBIDDEN_REPLACEMENTS = {
    "сейчас уточню": "Пришлите фото/модель — подскажу точнее.",
    "напишу через минуту": "Пришлите фото/модель — подскажу точнее.",
    "проверю и вернусь": "Пришлите фото/модель — подскажу точнее.",
}

app = FastAPI()
_avito_token_cache = {"token": None, "expires_at": 0}
chat_histories: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 12
DEFAULT_CLARIFICATION = "Пришлите фото/модель — подскажу точнее."
CLARIFICATION_ALTERNATIVES = [
    "Понял. Напишите модель или пришлите фото, и я точнее подскажу по вашему вопросу.",
    "Хорошо, давайте уточним по модели или фото — так не ошибёмся с ответом.",
]


def read_text_or_fallback(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        logging.warning("File is unavailable, using fallback: %s", path)
        return fallback.strip()


def build_system_prompt() -> str:
    system_prompt = read_text_or_fallback(SYSTEM_PROMPT_PATH, FALLBACK_SYSTEM_PROMPT)
    knowledge = read_text_or_fallback(KNOWLEDGE_PATH, FALLBACK_KNOWLEDGE)
    scripts = read_text_or_fallback(SCRIPTS_PATH, FALLBACK_SCRIPTS)
    offers = read_text_or_fallback(OFFERS_PATH, FALLBACK_OFFERS)
    mode = os.getenv("BOT_MODE", "AUTO").strip().upper()
    if mode not in {"AUTO", "ASSIST", "MANUAL"}:
        mode = "AUTO"

    return (
        f"{system_prompt}\n\n"
        f"## Текущий режим\n\n{mode}\n\n"
        f"## База знаний\n\n{knowledge}\n\n"
        f"## Готовые сценарии\n\n{scripts}\n\n"
        f"## Текущие акции и специальные предложения\n\n{offers}"
    )


def sanitize_reply(text: str) -> str:
    reply = text.strip()
    for forbidden, replacement in FORBIDDEN_REPLACEMENTS.items():
        reply = reply.replace(forbidden, replacement)
        reply = reply.replace(forbidden.capitalize(), replacement)
    return reply


def avoid_repeated_clarification(reply: str, history: list[dict[str, str]]) -> str:
    previous_assistant_messages = [
        item["content"].strip()
        for item in reversed(history)
        if item.get("role") == "assistant" and item.get("content")
    ]
    if not previous_assistant_messages:
        return reply

    previous_reply = previous_assistant_messages[0]
    if reply.strip() != DEFAULT_CLARIFICATION or previous_reply != DEFAULT_CLARIFICATION:
        return reply

    return CLARIFICATION_ALTERNATIVES[0]


def local_fallback_reply(text: str) -> str:
    lowered = text.lower()

    if "/start" in lowered or "привет" in lowered or "здравствуйте" in lowered:
        return "Здравствуйте! Подскажите, что нужно: заточка, профилирование, лезвия или доставка?"

    if "адрес" in lowered or "где" in lowered:
        return "Мы находимся: Новосибирск, ул. Богдана Хмельницкого, 27."

    if "профил" in lowered:
        return "Профилирование стоит 1290 ₽. При профилировании заточка входит в подарок."

    if "желоб" in lowered or "скольж" in lowered:
        return (
            "Чем больше цифра желоба, тем он более пологий: скольжение лучше, сопротивление меньше, "
            "зацеп и устойчивость ниже. Чем меньше цифра желоба, тем желоб глубже, зацеп выше, но скольжение хуже."
        )

    if "заточ" in lowered:
        return (
            "Классическая заточка — 400 ₽, новая пара — 590 ₽, фигурные — 490 ₽, "
            "ржавые / повреждённые — от 490 ₽."
        )

    if "лезв" in lowered or "base" in lowered or "pro" in lowered or "бюджет" in lowered:
        if "base" in lowered or "pro" in lowered or "бюджет" in lowered:
            return "Понял. Подскажите стакан и размер лезвия — так точно проверим подходящий вариант."
        return (
            "Подскажите, под какой стакан нужны лезвия: Bauer Tuuk Edge, Fly/PowerFly, "
            "Vertexx или CCM XS? И какой размер указан на старом лезвии — например 263, 272, 280?"
        )

    return "Пришлите фото/модель — подскажу точнее."


def generate_reply(buyer_message: str, chat_id: str = "") -> str:
    history_key = chat_id or "default"
    history = chat_histories.setdefault(history_key, [])

    if os.getenv("BOT_MODE", "AUTO").strip().upper() == "MANUAL":
        reply = avoid_repeated_clarification(DEFAULT_CLARIFICATION, history)
        history.append({"role": "user", "content": buyer_message})
        history.append({"role": "assistant", "content": reply})
        chat_histories[history_key] = history[-MAX_HISTORY_MESSAGES:]
        return reply

    if not ANTHROPIC_API_KEY:
        reply = sanitize_reply(local_fallback_reply(buyer_message))
        reply = avoid_repeated_clarification(reply, history)
        history.append({"role": "user", "content": buyer_message})
        history.append({"role": "assistant", "content": reply})
        chat_histories[history_key] = history[-MAX_HISTORY_MESSAGES:]
        return reply

    history.append({"role": "user", "content": buyer_message})
    recent_history = history[-MAX_HISTORY_MESSAGES:]

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            temperature=0.2,
            system=build_system_prompt(),
            messages=recent_history,
        )
        reply = response.content[0].text
    except Exception:
        logging.exception("LLM reply generation failed")
        reply = local_fallback_reply(buyer_message)

    reply = sanitize_reply(reply)
    reply = avoid_repeated_clarification(reply, history)
    history.append({"role": "assistant", "content": reply})
    chat_histories[history_key] = history[-MAX_HISTORY_MESSAGES:]
    return reply


async def get_avito_token() -> str:
    if not AVITO_CLIENT_ID or not AVITO_CLIENT_SECRET:
        raise RuntimeError("AVITO_CLIENT_ID and AVITO_CLIENT_SECRET are required")

    cached_token = _avito_token_cache.get("token")
    if cached_token and time.time() < float(_avito_token_cache.get("expires_at", 0)):
        return str(cached_token)

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            "https://api.avito.ru/token",
            data={
                "grant_type": "client_credentials",
                "client_id": AVITO_CLIENT_ID,
                "client_secret": AVITO_CLIENT_SECRET,
            },
        )
        response.raise_for_status()
        data = response.json()

    _avito_token_cache["token"] = data["access_token"]
    _avito_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
    return data["access_token"]


async def send_avito_message(user_id: int, chat_id: str, text: str) -> None:
    token = await get_avito_token()
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"https://api.avito.ru/messenger/v1/accounts/{user_id}/chats/{chat_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": {"text": text}, "type": "text"},
        )
        response.raise_for_status()


async def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN is not set")
        return

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        response.raise_for_status()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/avito")
async def avito_webhook(request: Request) -> dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    value = payload.get("payload", {}).get("value", {})
    if value.get("type") != "message":
        return {"status": "ignored"}

    chat_id = value.get("chat_id")
    user_id = value.get("user_id")
    author_id = value.get("author_id")
    text = (value.get("content", {}).get("text") or "").strip()

    if not chat_id or not user_id or not text:
        return {"status": "empty"}

    if str(author_id) == str(user_id):
        return {"status": "own_message"}

    reply = generate_reply(text, chat_id=f"avito_{chat_id}")
    await send_avito_message(user_id=user_id, chat_id=chat_id, text=reply)
    return {"status": "ok"}


@app.post("/webhook/telegram")
async def telegram_webhook(request: Request) -> dict[str, str]:
    payload = await request.json()
    message = payload.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return {"status": "ignored"}

    reply = generate_reply(text, chat_id=f"tg_{chat_id}")
    await send_telegram_message(chat_id, reply)
    return {"status": "ok"}
