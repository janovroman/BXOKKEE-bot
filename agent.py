import logging
import os
import re
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
DIALOGUE_EXAMPLES_PATH = BASE_DIR / "dialogue_examples.md"

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
Можем зарегистрировать вас в UDS — получите 500 приветственных бонусов и сможете копить бонусы на следующие покупки и услуги.
"""

FALLBACK_OFFERS = """# Текущие акции и специальные предложения

## Активные предложения

Пока активных специальных предложений нет.

## Правила

Все цены из knowledge.md считать базовыми. Если активной акции нет, не выдумывать скидку. UDS — отдельная бонусная система.
"""

FALLBACK_DIALOGUE_EXAMPLES = """Пример: если клиент уже указал стакан и размер лезвия, не просить фото и не подставлять цену от другого стакана.
"""

FORBIDDEN_REPLACEMENTS = {
    "сейчас уточню": "Пришлите фото/модель — подскажу точнее.",
    "напишу через минуту": "Пришлите фото/модель — подскажу точнее.",
    "проверю и вернусь": "Пришлите фото/модель — подскажу точнее.",
}

PRICE_TABLE = {
    "budget": {
        "Bauer Tuuk Edge": 3490,
        "Bauer Fly/PowerFly": 4990,
        "Bauer Vertexx": 4490,
        "CCM XS": 3490,
    },
    "base": {
        "Bauer Tuuk Edge": 4990,
        "Bauer Fly/PowerFly": 5990,
        "Bauer Vertexx": 5490,
        "CCM XS": 5490,
    },
    "pro": {
        "Bauer Tuuk Edge": 5990,
        "Bauer Fly/PowerFly": 7990,
        "Bauer Vertexx": 6990,
        "CCM XS": 6990,
    },
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
    dialogue_examples = read_text_or_fallback(DIALOGUE_EXAMPLES_PATH, FALLBACK_DIALOGUE_EXAMPLES)
    mode = os.getenv("BOT_MODE", "AUTO").strip().upper()
    if mode not in {"AUTO", "ASSIST", "MANUAL"}:
        mode = "AUTO"

    return (
        f"{system_prompt}\n\n"
        f"## Текущий режим\n\n{mode}\n\n"
        f"## База знаний\n\n{knowledge}\n\n"
        f"## Готовые сценарии\n\n{scripts}\n\n"
        f"## Примеры клиентских диалогов\n\n{dialogue_examples}\n\n"
        f"## Текущие акции и специальные предложения\n\n{offers}"
    )


def sanitize_reply(text: str) -> str:
    reply = text.strip().replace("**", "").replace("__", "")
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


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def user_history_text(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    user_messages = [
        item.get("content", "")
        for item in history[-MAX_HISTORY_MESSAGES:]
        if item.get("role") == "user"
    ]
    return normalize_text("\n".join(user_messages))


def has_edge(text: str) -> bool:
    return any(
        token in text
        for token in (
            "edge",
            "tuuk edge",
            "тук эдж",
            "туук эдж",
            "тук едж",
            "туук едж",
            "едже",
            "едж",
            "эдж",
        )
    )


def has_fly(text: str) -> bool:
    return any(token in text for token in ("powerfly", "power fly", "fly", "пауэрфлай", "флай"))


def has_ccm_xs(text: str) -> bool:
    has_ccm = "ccm" in text or "ццм" in text
    return "xs" in text and has_ccm


def has_xs(text: str) -> bool:
    return "xs" in text


def has_vertexx(text: str) -> bool:
    return "vertexx" in text or "вертекс" in text


def has_goalie_bauer(text: str) -> bool:
    return (
        ("вратар" in text and ("bauer" in text or "бауер" in text))
        or "bauer goalie" in text
        or "goalie bauer" in text
        or "бауер goalie" in text
        or "goalie бауер" in text
    )


def extract_vertexx_size(text: str) -> str | None:
    for match in re.findall(r"\b(?:размер\s*)?([1-9]|1[0-2])\b", text):
        return match
    return None


def extract_blade_size(text: str) -> str | None:
    for match in re.findall(r"\b\d{3}\b", text):
        size = int(match)
        if 200 <= size <= 330:
            return match
    return None


def has_blade_request(text: str) -> bool:
    return "лезв" in text


def has_budget_request(text: str) -> bool:
    return any(
        token in text
        for token in (
            "подешевле",
            "дешевле",
            "бюджет",
            "самые недорогие",
            "самый недорогой",
            "самый доступный",
            "более дешевые",
            "более дешевый",
        )
    )


def has_base_request(text: str) -> bool:
    return "base" in text or "бейс" in text


def has_pro_request(text: str) -> bool:
    return (
        re.search(r"\bpro\b", text) is not None
        or re.search(r"(^|\s)про($|\s)", text) is not None
        or "лучше по ресурсу" in text
        or "лучший по ресурсу" in text
        or "самый лучший" in text
        or "ресурс" in text
    )


def price_for(line: str, holder: str) -> int | None:
    return PRICE_TABLE.get(line, {}).get(holder)


def price_missing_reply() -> str:
    return "По цене для этого стакана лучше уточним по наличию."


def blade_options_reply(holder: str, size: str) -> str:
    return (
        f"Понял, нужны лезвия {holder} {size}.\n\n"
        "Есть три варианта:\n"
        "1. Бюджетные лезвия без бренда\n"
        "2. in hockey® Base\n"
        "3. in hockey® Pro\n\n"
        "Какой вариант рассматриваете — самый доступный или лучше по ресурсу?"
    )


def budget_blade_reply(holder: str, size: str) -> str:
    price = price_for("budget", holder)
    if price is None:
        return price_missing_reply()
    return (
        f"Бюджетные лезвия без бренда для {holder} {size} — базовая цена {price} ₽.\n\n"
        "Можем зарегистрировать вас в UDS — получите 500 приветственных бонусов и сможете копить бонусы на следующие покупки и услуги."
    )


def base_blade_reply(holder: str, size: str) -> str:
    price = price_for("base", holder)
    if price is None:
        return price_missing_reply()
    return (
        f"in hockey® Base для {holder} {size} — базовая цена {price} ₽.\n\n"
        "Можем зарегистрировать вас в UDS — получите 500 приветственных бонусов и сможете копить бонусы на следующие покупки и услуги."
    )


def pro_blade_reply(holder: str, size: str) -> str:
    price = price_for("pro", holder)
    if price is None:
        return price_missing_reply()
    return (
        f"in hockey® Pro для {holder} {size} — базовая цена {price} ₽.\n\n"
        "Это старшая линейка: лучший ресурс, обработка и база под индивидуальное профилирование.\n\n"
        "Можем зарегистрировать вас в UDS — получите 500 приветственных бонусов и сможете копить бонусы на следующие покупки и услуги."
    )


def goalie_bauer_reply() -> str:
    return (
        "Скорее всего, у вас стакан Bauer Vertexx. Подскажите размер, который указан "
        "на старом лезвии, или пришлите фото стакана — там на пяточной области есть цифра. Подберём точнее."
    )


def goalie_bauer_size_reply() -> str:
    return (
        "Понял, речь про вратарские Bauer. Скорее всего, это стакан Bauer Vertexx. "
        "Уточните, пожалуйста, размер именно на старом лезвии или пришлите фото стакана — "
        "там на пяточной области есть цифра. Так не ошибёмся."
    )


def known_holder_and_size(full_context: str) -> tuple[str, str] | None:
    size = extract_blade_size(full_context)
    if has_ccm_xs(full_context) and size:
        return "CCM XS", size
    if has_edge(full_context) and size and ("bauer" in full_context or "бауер" in full_context or has_blade_request(full_context)):
        return "Bauer Tuuk Edge", size
    if has_fly(full_context) and size and ("bauer" in full_context or "бауер" in full_context or has_blade_request(full_context)):
        return "Bauer Fly/PowerFly", size
    vertexx_size = extract_vertexx_size(full_context)
    if has_vertexx(full_context) and vertexx_size:
        return "Bauer Vertexx", vertexx_size
    if has_xs(full_context) and size:
        return "CCM XS", size
    return None


def local_fallback_reply(text: str, history: list[dict[str, str]] | None = None) -> str:
    lowered = normalize_text(text)
    context = user_history_text(history)
    full_context = f"{context}\n{lowered}".strip()
    holder_size = known_holder_and_size(full_context)
    current_holder_size = known_holder_and_size(lowered)
    has_bauer = "bauer" in full_context or "бауер" in full_context
    current_has_bauer = "bauer" in lowered or "бауер" in lowered
    full_has_edge = has_edge(full_context)
    current_has_edge = has_edge(lowered)
    current_has_vertexx = has_vertexx(lowered)
    current_goalie_bauer = has_goalie_bauer(lowered)
    current_vertexx_size = extract_vertexx_size(lowered)
    blade_context = has_blade_request(full_context) or has_bauer or full_has_edge or has_vertexx(full_context) or has_ccm_xs(full_context)

    if current_goalie_bauer and "размер" in lowered and current_vertexx_size and not current_has_vertexx:
        return goalie_bauer_size_reply()

    if current_goalie_bauer and not current_has_vertexx:
        return goalie_bauer_reply()

    if has_budget_request(lowered):
        if holder_size:
            holder, size = holder_size
            return budget_blade_reply(holder, size)
        return "Самый доступный вариант — бюджетные лезвия без бренда. Для точной проверки подскажите стакан и размер лезвия."

    if has_base_request(lowered) and holder_size:
        holder, size = holder_size
        return base_blade_reply(holder, size)

    if has_pro_request(lowered) and holder_size:
        holder, size = holder_size
        return pro_blade_reply(holder, size)

    if current_holder_size:
        holder, size = current_holder_size
        return blade_options_reply(holder, size)

    if holder_size and (has_blade_request(lowered) or current_has_edge or current_has_vertexx or has_xs(lowered) or extract_blade_size(lowered)):
        holder, size = holder_size
        return blade_options_reply(holder, size)

    if blade_context and (has_bauer or current_has_edge) and current_has_edge and not extract_blade_size(full_context):
        return "Понял, Bauer Tuuk Edge. Какой размер лезвия нужен? Например 263, 272, 280."

    if blade_context and current_has_bauer and not (full_has_edge or has_fly(full_context) or has_vertexx(full_context)):
        return "По Bauer уточните стакан: Tuuk Edge, Fly/PowerFly или Vertexx?"

    if has_xs(lowered) and not extract_blade_size(full_context):
        return "Понял, CCM XS. Какой размер лезвия нужен? Например 263, 271, 280."

    if has_blade_request(lowered):
        return (
            "Подскажите, под какой стакан нужны лезвия: Bauer Tuuk Edge, Fly/PowerFly, "
            "Vertexx или CCM XS? И какой размер указан на старом лезвии — например 263, 272, 280?"
        )

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
        recent_history = [*history, {"role": "user", "content": buyer_message}][-MAX_HISTORY_MESSAGES:]
        reply = sanitize_reply(local_fallback_reply(buyer_message, recent_history))
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
        reply = local_fallback_reply(buyer_message, recent_history)

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
