import re

SENTENCE_END_PATTERN = re.compile(r"[.!?]+\s+")


class SentenceChunker:
    def __init__(self):
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        self._buffer += token
        sentences = []

        while True:
            match = SENTENCE_END_PATTERN.search(self._buffer)
            if not match:
                break

            sentence = self._buffer[:match.end()].strip()
            self._buffer = self._buffer[match.end():]

            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None