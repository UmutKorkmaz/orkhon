"""Deterministic specialist routing for the unified Orkhon assistant.

Small LLMs are the wrong place to perform exact symbolic transforms. The unified
assistant therefore keeps deterministic tasks in code and leaves open-ended
language behavior to the model.
"""

from __future__ import annotations

import re

from orkhon.data.old_turkic import OLD_TURKIC_RANGE, rune_to_latin


_ADD_RE = re.compile(
    r"(?:what\s+is\s+)?(?P<a>\d{1,4})\s*\+\s*(?P<b>\d{1,4})(?:\s*(?:\?|kactir|kaçtır))?",
    re.IGNORECASE,
)

_TR_MARKERS = (
    "merhaba",
    "selam",
    "nasılsın",
    "nasilsin",
    "hikaye",
    "yardım",
    "yardim",
    "anlat",
    "yaz",
    "türkçe",
    "turkce",
)


def _is_old_turkic(ch: str) -> bool:
    cp = ord(ch)
    return OLD_TURKIC_RANGE[0] <= cp <= OLD_TURKIC_RANGE[1]


def _extract_old_turkic(text: str) -> str:
    parts: list[str] = []
    in_gap = False
    for ch in text:
        if _is_old_turkic(ch):
            if in_gap and parts and parts[-1] != " ":
                parts.append(" ")
            parts.append(ch)
            in_gap = False
        elif ch.isspace() and parts:
            in_gap = True
        elif parts:
            in_gap = True
    return "".join(parts).strip()


def _looks_like_translation_request(text: str) -> bool:
    lower = text.lower()
    translation_terms = (
        "translate",
        "translation",
        "modern turkish",
        "modern english",
        "ceviri",
        "çeviri",
        "tercüme",
        "tercume",
        "turkceye",
        "türkçeye",
        "ingilizceye",
    )
    transliteration_terms = (
        "transliterate",
        "transliteration",
        "latin",
        "read",
        "oku",
        "harflerine",
        "translitere",
    )
    return any(t in lower for t in translation_terms) and not any(
        t in lower for t in transliteration_terms
    )


def _is_transliteration_request(text: str) -> bool:
    lower = text.lower()
    terms = (
        "old turkic",
        "kokturk",
        "gokturk",
        "göktürk",
        "eski turkce",
        "eski türkçe",
        "rune",
        "runic",
        "transliterate",
        "transliteration",
        "latin",
        "harflerine",
        "cevir",
        "çevir",
        "oku",
        "read",
    )
    return any(t in lower for t in terms)


def _math_reply(text: str) -> str | None:
    match = _ADD_RE.search(text)
    if not match:
        return None
    a = int(match.group("a"))
    b = int(match.group("b"))
    lower = text.lower()
    if "kactir" in lower or "kaçtır" in lower:
        return f"Cevap {a + b}."
    return f"The answer is {a + b}."


def _is_turkish(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _TR_MARKERS) or any(
        ch in lower for ch in "çğıöşü"
    )


def _general_reply(text: str) -> str | None:
    lower = text.lower().strip()
    compact = re.sub(r"\s+", " ", lower)
    turkish = _is_turkish(text)

    if re.fullmatch(r"(test|deneme|ping)[.!?]*", compact):
        if turkish:
            return "Test çalışıyor. Orkhon cevap verebiliyor."
        return "Test is working. Orkhon can respond."

    if re.fullmatch(r"(merhaba|selam|selamlar|hello|hi|hey)[.!?]*", compact):
        if turkish:
            return (
                "Merhaba. Türkçe ve İngilizce soruları yanıtlayabilir, kısa "
                "açıklamalar yapabilir, basit hesapları çözebilir ve Göktürk/"
                "Eski Türkçe runelerini Latin harflerine çevirebilirim."
            )
        return (
            "Hello. I can answer in English and Turkish, give short explanations, "
            "handle simple arithmetic, and transliterate Old Turkic/Kokturk runes "
            "into Latin letters."
        )

    if "nasılsın" in lower or "nasilsin" in lower or "how are you" in lower:
        if turkish:
            return (
                "İyiyim; Orkhon olarak kısa ve net yardımcı olmak için buradayım. "
                "Bir soru sorabilir, metin özetletebilir ya da Göktürk runelerini "
                "Latin harflerine çevirmemi isteyebilirsin."
            )
        return (
            "I am working normally. Ask a question, request a short explanation, "
            "or send Old Turkic/Kokturk runes for Latin transliteration."
        )

    if (
        "what can you help" in lower
        or "what can you do" in lower
        or "ne yapabilirsin" in lower
        or "hangi konularda" in lower
        or "yardım edebilirsin" in lower
        or "yardim edebilirsin" in lower
    ):
        if turkish:
            return (
                "Türkçe ve İngilizce soruları yanıtlayabilir, kısa açıklama ve "
                "özet yazabilir, basit hesapları çözebilir ve Göktürk/Eski Türkçe "
                "runelerini Latin harflerine çevirebilirim. Güvenilir çeviri için "
                "kaynaklı yazıt verisi gerekir; anlam uydurmam."
            )
        return (
            "I can answer English and Turkish questions, write short explanations "
            "and summaries, solve simple arithmetic, and transliterate Old Turkic/"
            "Kokturk runes into Latin. Reliable inscription translation needs "
            "sourced data, so I should not invent meanings."
        )

    wants_story = (
        ("hikaye" in lower or "masal" in lower)
        and any(word in lower for word in ("anlat", "yaz", "söyle", "soyle"))
    ) or "tell me a story" in lower or "write a story" in lower
    if wants_story:
        if turkish:
            return (
                "Kısa hikaye: Bozkırda genç bir yazıcı, rüzgarın sildiği izleri "
                "taşa kazımayı öğrenmiş. Her harfi acele etmeden işlemiş; çünkü "
                "biliyormuş ki söz uçarsa bile doğru yazılan iz kalır. Gün "
                "batarken son satıra şunu eklemiş: 'Bilgi, paylaşıldığında yol olur.'"
            )
        return (
            "Short story: A young scribe crossed the steppe carrying only a small "
            "knife and a memory of old words. When the wind erased every footprint, "
            "the scribe carved the lesson into stone: knowledge becomes a road when "
            "it is shared."
        )

    return None


def deterministic_reply(message: str) -> str | None:
    """Return an exact tool-style reply when the message matches a known task.

    Supported tasks:
    - Old Turkic/Kokturk rune to Latin transliteration.
    - Cautious response for translation requests involving runes.
    - Simple integer addition.
    """
    text = (message or "").strip()
    if not text:
        return None

    runic = _extract_old_turkic(text)
    if runic and _looks_like_translation_request(text):
        transliteration = rune_to_latin(runic)
        return (
            "I can transliterate these Old Turkic/Kokturk runes into Latin, but "
            "reliable translation to modern Turkish or English needs sourced "
            f"inscription data. Latin transliteration: {transliteration}"
        )
    if runic and _is_transliteration_request(text):
        return rune_to_latin(runic)

    math = _math_reply(text)
    if math is not None:
        return math
    return _general_reply(text)
