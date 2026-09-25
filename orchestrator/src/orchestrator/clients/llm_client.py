import time

import requests

SYSTEM_PROMPT = (
    "You are a voice assistant. Your replies are converted to speech and read "
    "aloud, so respond in plain spoken language only. Do not use markdown, "
    "asterisks, bullet points, headers, code blocks, or emojis. Do not use "
    "any symbols that would sound strange if read aloud."
)


def chat(text: str, uri: str, max_retries: int = 5, retry_delay_seconds: float = 2.0) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = requests.post(uri, json=payload, timeout=120)
            if response.status_code == 503:
                last_error = RuntimeError("llm service not ready yet")
                time.sleep(retry_delay_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            time.sleep(retry_delay_seconds)

    raise RuntimeError(f"llm service unavailable after {max_retries} retries") from last_error