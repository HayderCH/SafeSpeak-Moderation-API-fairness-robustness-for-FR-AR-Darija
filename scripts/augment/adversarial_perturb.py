"""Generate adversarially perturbed variants of SafeSpeak datasets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml

from safespeak.augment.perturbations import apply_recipes_sequence

DEFAULT_WORKFLOW_TAG = "syn-002-v0"
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_MAX_VARIANTS = 2
RANDOM_SOURCE_ID_COLS = ["source_id", "id", "comment_id", "example_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate adversarial perturbations for SafeSpeak datasets."),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration describing inputs and recipes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalise_input_specs(raw_specs: Iterable[Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for entry in raw_specs:
        if isinstance(entry, str):
            specs.append({"path": entry})
        elif isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError("Each input entry must include a 'path'.")
            specs.append(entry)
        else:
            raise TypeError("Input entries must be strings or dictionaries.")
    return specs


def _load_inputs(specs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for entry in specs:
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")
        df = pd.read_csv(path)
        max_rows = entry.get("max_rows")
        if max_rows is not None:
            df = df.head(int(max_rows)).copy()
        df["__source_dataset"] = entry.get("source_dataset", path.stem)
        df["__source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No input datasets loaded.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _ensure_text_column(df: pd.DataFrame, text_column: str) -> None:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}' in inputs.")
    if df[text_column].isnull().any():
        df[text_column] = df[text_column].fillna("")


def _derive_source_ids(df: pd.DataFrame) -> None:
    for col in RANDOM_SOURCE_ID_COLS:
        if col in df.columns:
            series = df[col].astype(str)
            if series.notna().any():
                df["__source_id"] = series.fillna("")
                df.loc[df["__source_id"] == "nan", "__source_id"] = ""
                df.loc[df["__source_id"] == "", "__source_id"] = df.index.map(str)
                return
    df["__source_id"] = df.index.map(str)


def _filter_df(
    df: pd.DataFrame,
    language_column: str,
    label_column: str,
    languages: Iterable[str] | None,
    labels: Iterable[str] | None,
) -> pd.DataFrame:
    result = df
    if languages:
        if language_column not in df.columns:
            raise ValueError(
                f"Language column '{language_column}' not found for filtering."
            )
        result = result[result[language_column].isin(languages)]
    if labels:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found for filtering.")
        result = result[result[label_column].isin(labels)]
    return result.copy()


def _prepare_sequence(
    config_recipes: List[Dict[str, Any]],
) -> List[Tuple[str, float, Dict[str, float]]]:
    sequence: List[Tuple[str, float, Dict[str, float]]] = []
    for recipe in config_recipes:
        name = recipe.get("name")
        if not name:
            raise ValueError("Recipe entries must include a 'name'.")
        probability = float(recipe.get("probability", 1.0))
        params = recipe.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Recipe 'params' must be a mapping.")
        sequence.append((name, probability, params))
    return sequence


def _generate_variants_for_row(
    row: pd.Series,
    sequence: Sequence[Tuple[str, float, Dict[str, float]]],
    rng: random.Random,
    max_variants: int,
    max_attempts: int,
    text_column: str,
    workflow_tag: str,
) -> List[Dict[str, Any]]:
    original_text = str(row[text_column])
    variants: List[Dict[str, Any]] = []
    seen_texts = {original_text}
    attempts = 0
    while len(variants) < max_variants and attempts < max_attempts:
        attempts += 1
        seed = rng.randrange(0, 2**32 - 1)
        local_rng = random.Random(seed)
        perturbed, applied = apply_recipes_sequence(
            original_text,
            local_rng,
            sequence,
        )
        if not applied or perturbed in seen_texts:
            continue
        seen_texts.add(perturbed)
        record = row.to_dict()
        record[text_column] = perturbed
        record.setdefault("source_text", original_text)
        record["adv_original_text"] = original_text
        record["adv_recipes"] = "|".join(applied)
        record["adv_seed"] = seed
        record["source_dataset"] = row["__source_dataset"]
        record["source_path"] = row["__source_path"]
        record["source_id"] = row["__source_id"]
        record["workflow_tag"] = workflow_tag
        variants.append(record)
    return variants


def _dedupe_variants(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_column])


def _split_eval(
    df: pd.DataFrame,
    eval_ratio: float,
    rng: random.Random,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if eval_ratio <= 0:
        return df, pd.DataFrame(columns=df.columns)
    shuffled = df.sample(frac=1.0, random_state=rng.randint(0, 2**32 - 1))
    cutoff = int(len(shuffled) * (1 - eval_ratio))
    train_df = shuffled.iloc[:cutoff].copy()
    eval_df = shuffled.iloc[cutoff:].copy()
    return train_df, eval_df


def run(config: Dict[str, Any], seed_override: int | None) -> None:
    sequence = _prepare_sequence(config.get("recipes", []))
    if not sequence:
        raise ValueError("Config must include at least one recipe.")

    rng_seed = (
        seed_override
        if seed_override is not None
        else int(config.get("seed", random.randrange(0, 2**32 - 1)))
    )
    rng = random.Random(rng_seed)

    input_specs = _normalise_input_specs(
        config.get("input_sources", config.get("input_paths", []))
    )
    if not input_specs:
        raise ValueError("Config must specify 'input_sources' or 'input_paths'.")

    df = _load_inputs(input_specs)

    text_column = config.get("text_column", "text")
    language_column = config.get("language_column", "language")
    label_column = config.get("label_column", "label")

    _ensure_text_column(df, text_column)
    _derive_source_ids(df)

    languages = config.get("language_filter")
    labels = config.get("label_filter")
    df = _filter_df(df, language_column, label_column, languages, labels)

    if df.empty:
        raise ValueError("No rows remain after filtering. Adjust config filters.")

    max_variants = int(config.get("max_variants_per_input", DEFAULT_MAX_VARIANTS))
    max_attempts = int(config.get("max_attempts_per_input", DEFAULT_MAX_ATTEMPTS))
    workflow_tag = config.get("workflow_tag", DEFAULT_WORKFLOW_TAG)

    output_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        variants = _generate_variants_for_row(
            row,
            sequence,
            rng,
            max_variants,
            max_attempts,
            text_column,
            workflow_tag,
        )
        output_records.extend(variants)

    if not output_records:
        raise RuntimeError(
            "No adversarial variants generated; adjust recipe probabilities."
        )

    result_df = pd.DataFrame(output_records)
    result_df = _dedupe_variants(result_df, text_column)
    internal_cols = [col for col in result_df.columns if col.startswith("__")]
    if internal_cols:
        result_df = result_df.drop(columns=internal_cols)

    eval_ratio = float(config.get("eval_split_ratio", 0.0))
    train_df, eval_df = _split_eval(result_df, eval_ratio, rng)

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)

    eval_output_path = config.get("eval_output_path")
    if eval_ratio > 0:
        if eval_output_path:
            eval_path = Path(eval_output_path)
        else:
            eval_path = output_path.with_name(output_path.stem + "_eval.csv")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(eval_path, index=False)

    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "seed": rng_seed,
        "config": config,
        "generated_rows": len(result_df),
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "text_column": text_column,
        "language_filter": languages,
        "label_filter": labels,
        "workflow_tag": workflow_tag,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(
        "Generated adversarial dataset:",
        f"total={len(result_df)}",
        f"train={len(train_df)}",
        f"eval={len(eval_df)}",
        f"output={output_path}",
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run(config, seed_override=args.seed)
