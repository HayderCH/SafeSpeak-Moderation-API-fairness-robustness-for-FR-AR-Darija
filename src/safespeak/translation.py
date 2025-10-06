"""Utility helpers for machine translation tasks.

This module centralizes translation helpers so that multiple scripts can share
common logic (e.g., dataset translation and back-translation utilities).
"""

from __future__ import annotations

import importlib
import os
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_MAP: Dict[Tuple[str, str], str] = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
}


def resolve_device(device_str: str, torch_mod) -> int:
    """Resolve a user-specified device string into a CUDA index or CPU flag."""

    option = device_str.lower()
    if option == "auto":
        return 0 if torch_mod.cuda.is_available() else -1
    if option == "cpu":
        return -1
    if option.startswith("cuda"):
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system")
        if option == "cuda":
            index = 0
        elif ":" in option:
            index = int(option.split(":", 1)[1])
        else:
            index = 0
        if index >= torch_mod.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} out of range")
        return index
    raise ValueError(f"Unsupported device option: {device_str}")


def infer_model(
    source_lang: str,
    target_lang: str,
    *,
    model_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> str:
    """Infer the Hugging Face model to use for a language pair."""

    model_map = model_map or MODEL_MAP
    key = (source_lang.lower(), target_lang.lower())
    if key not in model_map:
        message = (
            "No translation model configured for " f"{source_lang}->{target_lang}."
        )
        raise ValueError(message)
    return model_map[key]


class Translator:
    """Lightweight wrapper around a MarianMT model for batch translation."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_str = device
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_imports(self) -> None:
        if self._torch is not None:
            return

        import_utils_mod = importlib.import_module("transformers.utils.import_utils")
        if getattr(import_utils_mod, "_torchvision_available", False):
            setattr(
                import_utils_mod,
                "_torchvision_available",
                False,
            )  # type: ignore[attr-defined]
            setattr(
                import_utils_mod,
                "_torchvision_version",
                "0",
            )  # type: ignore[attr-defined]

        try:
            transformers_mod = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required. Install with "
                "`pip install -e .[translation]`."
            ) from exc

        try:
            torch_mod = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "torch is required. Install with " "`pip install -e .[translation]`."
            ) from exc

        device_index = resolve_device(self.device_str, torch_mod)
        device = (
            torch_mod.device(f"cuda:{device_index}")
            if device_index >= 0
            else torch_mod.device("cpu")
        )

        tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self.model_name)
        model = transformers_mod.AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()

        self._torch = torch_mod
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def translate(
        self,
        texts: Iterable[str],
        *,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> List[str]:
        """Translate a sequence of texts, returning a list of outputs."""

        self._ensure_imports()
        assert self._torch is not None  # for type checkers
        assert self._tokenizer is not None
        assert self._model is not None

        text_list = list(texts)
        outputs: List[str] = []
        iterator = range(0, len(text_list), self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError:  # pragma: no cover - optional dependency
                tqdm = None  # type: ignore[assignment]
            else:
                iterator = tqdm(iterator, desc=progress_desc or "Translating")

        for start in iterator:
            batch = text_list[start : start + self.batch_size]
            if not batch:
                continue
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_length=512,
                )
            decoded = self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            outputs.extend(decoded)
        return outputs


def translate_texts(
    texts: Iterable[str],
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: int = 16,
    device: str = "auto",
    model_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[str]:
    """Translate a collection of texts using a configured MarianMT model."""

    if model_name is None:
        if source_lang is None or target_lang is None:
            raise ValueError(
                "Either model_name or source/target languages must be " "provided."
            )
        model_name = infer_model(source_lang, target_lang, model_map=model_map)

    translator = Translator(model_name, batch_size=batch_size, device=device)
    return translator.translate(texts)
