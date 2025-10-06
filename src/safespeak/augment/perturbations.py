"""Text perturbation primitives for adversarial data augmentation."""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Tuple

from unidecode import unidecode

# Mapping of letters to keyboard-adjacent alternatives (simplified QWERTY).
_KEYBOARD_NEIGHBORS: Dict[str, str] = {
    "a": "qsxz",
    "b": "vghn",
    "c": "xdfv",
    "d": "sefx",
    "e": "wsdr",
    "f": "drtg",
    "g": "ftyhb",
    "h": "gyujn",
    "i": "ujko",
    "j": "huikm",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedx",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "xsa",
}

_LEET_MAP: Dict[str, str] = {
    "a": "4",
    "b": "8",
    "e": "3",
    "g": "9",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
    "t": "7",
}

_ZERO_WIDTH_CHARS: Tuple[str, ...] = ("\u200b", "\u200c", "\u200d")

_EMOJI_CHOICES: Tuple[str, ...] = (
    "😈",
    "🤬",
    "🔥",
    "💅",
    "😤",
    "😡",
    "👿",
    "🙄",
    "😂",
    "🤣",
)

PerturbFn = Callable[[str, random.Random, Dict[str, float]], Tuple[str, bool]]


def _swap_characters(text: str, idx: int) -> str:
    if idx < 0 or idx >= len(text) - 1:
        return text
    chars = list(text)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _replace_with_neighbor(char: str, rng: random.Random) -> str:
    lower = char.lower()
    neighbors = _KEYBOARD_NEIGHBORS.get(lower)
    if not neighbors:
        return char
    substitute = rng.choice(neighbors)
    return substitute.upper() if char.isupper() else substitute


def typo_noise(
    text: str,
    rng: random.Random,
    params: Dict[str, float],
) -> Tuple[str, bool]:
    """Introduce keyboard-typo style noise."""

    swap_prob = params.get("swap_prob", 0.2)
    delete_prob = params.get("delete_prob", 0.1)
    substitute_prob = params.get("substitute_prob", 0.3)

    changed = False
    chars: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if rng.random() < delete_prob:
            changed = True
            i += 1
            continue
        if rng.random() < swap_prob:
            swapped = _swap_characters(text, i)
            if swapped != text:
                text = swapped
                changed = True
                i += 2
                continue
        if rng.random() < substitute_prob and char.isalpha():
            replacement = _replace_with_neighbor(char, rng)
            if replacement != char:
                chars.append(replacement)
                changed = True
            else:
                chars.append(char)
        else:
            chars.append(char)
        i += 1

    result = "".join(chars)
    return (result, changed and result != text)


def leetspeak_noise(
    text: str,
    rng: random.Random,
    params: Dict[str, float],
) -> Tuple[str, bool]:
    """Apply leetspeak substitutions to characters."""

    prob = params.get("probability", 0.25)
    chars: List[str] = []
    changed = False
    for char in text:
        lower = char.lower()
        if lower in _LEET_MAP and rng.random() < prob:
            replacement = _LEET_MAP[lower]
            chars.append(replacement)
            changed = True
        else:
            chars.append(char)
    result = "".join(chars)
    return result, changed and result != text


def spacing_noise(
    text: str,
    rng: random.Random,
    params: Dict[str, float],
) -> Tuple[str, bool]:
    """Adjust spacing and insert zero-width characters."""

    drop_prob = params.get("drop_space_prob", 0.15)
    duplicate_prob = params.get("duplicate_space_prob", 0.1)
    zero_width_prob = params.get("zero_width_prob", 0.1)

    result_chars: List[str] = []
    changed = False
    for char in text:
        if char == " ":
            if rng.random() < drop_prob:
                changed = True
                continue
            result_chars.append(char)
            if rng.random() < duplicate_prob:
                result_chars.append(char)
                changed = True
        else:
            result_chars.append(char)
        if rng.random() < zero_width_prob:
            result_chars.append(rng.choice(_ZERO_WIDTH_CHARS))
            changed = True
    result = "".join(result_chars)
    return result, changed and result != text


def emoji_noise(
    text: str,
    rng: random.Random,
    params: Dict[str, float],
) -> Tuple[str, bool]:
    """Inject emojis near high-salience tokens."""

    insertion_prob = params.get("insertion_prob", 0.2)
    append_prob = params.get("append_prob", 0.3)

    tokens = text.split()
    if not tokens:
        return text, False

    changed = False
    for idx, token in enumerate(tokens):
        if rng.random() < insertion_prob:
            emoji = rng.choice(_EMOJI_CHOICES)
            tokens[idx] = f"{token}{emoji}"
            changed = True
    result = " ".join(tokens)
    if rng.random() < append_prob:
        emoji = rng.choice(_EMOJI_CHOICES)
        result = f"{result} {emoji}"
        changed = True
    return result, changed and result != text


def unicode_confusables(
    text: str,
    rng: random.Random,
    params: Dict[str, float],
) -> Tuple[str, bool]:
    """Replace ASCII with visually similar Unicode characters."""

    prob = params.get("probability", 0.2)
    changed = False
    chars: List[str] = []
    for char in text:
        if char.isalpha() and rng.random() < prob:
            ascii_char = unidecode(char.lower())
            if ascii_char and ascii_char != char:
                chars.append(ascii_char)
                changed = True
            else:
                chars.append(char)
        else:
            chars.append(char)
    result = "".join(chars)
    return result, changed and result != text


AVAILABLE_RECIPES: Dict[str, PerturbFn] = {
    "typo": typo_noise,
    "leetspeak": leetspeak_noise,
    "spacing": spacing_noise,
    "emoji": emoji_noise,
    "confusable": unicode_confusables,
}


class RecipeApplicationError(RuntimeError):
    """Raised when an unknown recipe is requested."""


def apply_recipes_sequence(
    text: str,
    rng: random.Random,
    sequence: Iterable[Tuple[str, float, Dict[str, float]]],
) -> Tuple[str, List[str]]:
    """Apply a sequence of perturbation recipes with probabilities.

    Args:
        text: Source text to perturb.
        rng: Random number generator.
        sequence: Iterable of tuples ``(name, probability, params)``.

    Returns:
        A tuple of ``(perturbed_text, applied_recipes)``.
    """

    current = text
    applied: List[str] = []
    for name, probability, params in sequence:
        if rng.random() > probability:
            continue
        recipe = AVAILABLE_RECIPES.get(name)
        if recipe is None:
            raise RecipeApplicationError(f"Unknown recipe: {name}")
        updated, changed = recipe(current, rng, params)
        if changed:
            current = updated
            applied.append(name)
    return current, applied
