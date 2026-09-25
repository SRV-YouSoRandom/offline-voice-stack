import requests

SYSTEM_PROMPT = (
    "You are a voice assistant. Your replies are converted to speech and read "
    "aloud, so respond in plain spoken language only. Do not use markdown, "
    "asterisks, bullet points, headers, code blocks, or emojis. Do not use "
    "any symbols that would sound strange if read aloud."
)


def chat(text: str, uri: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(uri, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]