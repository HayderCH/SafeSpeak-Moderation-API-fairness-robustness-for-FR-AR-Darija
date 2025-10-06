"""Text normalization utilities for multilingual toxicity detection."""

from __future__ import annotations

import re
from collections.abc import Iterable

import emoji
from unidecode import unidecode

# Basic leetspeak mapping, extendable via configuration
LEET_MAPPING = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
}

EMOJI_REPLACEMENTS = {
    "🔥": "anger",
    "💀": "dead",
    "🤢": "disgust",
    "🤮": "disgust",
    "😡": "angry",
    "😠": "angry",
    "😂": "laugh",
    "🤣": "laugh",
    "😭": "cry",
    "👊": "punch",
    "💣": "bomb",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
MULTIPLE_SPACES = re.compile(r"\s+")


def strip_urls(text: str) -> str:
    """Remove URLs from text."""
    return URL_PATTERN.sub(" ", text)


def strip_user_handles(text: str) -> str:
    """Replace user handles with a placeholder."""
    return USER_PATTERN.sub(" <user> ", text)


def normalize_hashtags(text: str) -> str:
    """Split hashtags into words when possible."""

    def _split_hashtag(match: re.Match[str]) -> str:
        hashtag = match.group()[1:]
        if hashtag.isupper():
            return f" {hashtag.lower()} "
        components = re.findall(r"[A-Z]?[a-z]+|[0-9]+", hashtag)
        if components:
            return " " + " ".join(c.lower() for c in components) + " "
        return " " + hashtag.lower() + " "

    return HASHTAG_PATTERN.sub(_split_hashtag, text)


def normalize_repeated_chars(text: str) -> str:
    """Limit elongated character sequences (e.g. coooool -> coool)."""
    return REPEATED_CHAR_PATTERN.sub(r"\1\1", text)


def map_leetspeak(text: str) -> str:
    """Convert common leetspeak characters to their alphabetic form."""
    return "".join(LEET_MAPPING.get(char.lower(), char) for char in text)


def replace_emojis(text: str) -> str:
    """Replace frequent emojis with textual descriptions; demojize the rest."""

    def _replace(char: str) -> str:
        if char in EMOJI_REPLACEMENTS:
            return f" {EMOJI_REPLACEMENTS[char]} "
        if char in emoji.EMOJI_DATA:
            description = emoji.demojize(char).strip(":")
            return " " + description.replace("_", " ") + " "
        return char

    return "".join(_replace(ch) for ch in text)


def normalize_text(
    text: str,
    *,
    strip_handles: bool = True,
    strip_links: bool = True,
    normalize_hashtag: bool = True,
    lowercase: bool = True,
    transliterate: bool = True,
    emoji_replace: bool = True,
    leetspeak: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Run SafeSpeak normalization pipeline on raw text."""
    processed = text
    if strip_links:
        processed = strip_urls(processed)
    if strip_handles:
        processed = strip_user_handles(processed)
    if normalize_hashtag:
        processed = normalize_hashtags(processed)
    if emoji_replace:
        processed = replace_emojis(processed)
    processed = normalize_repeated_chars(processed)
    if leetspeak:
        processed = map_leetspeak(processed)
    if transliterate:
        processed = unidecode(processed)
    if lowercase:
        processed = processed.lower()
    if collapse_whitespace:
        processed = MULTIPLE_SPACES.sub(" ", processed).strip()
    return processed


def batch_normalize(texts: Iterable[str], **kwargs) -> list[str]:
    """Normalize an iterable of texts."""
    return [normalize_text(text, **kwargs) for text in texts]
