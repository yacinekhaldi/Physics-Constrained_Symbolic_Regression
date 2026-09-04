from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_dirs

DATASET_ID = "5b5n3cg32n"
DATASET_VERSION = 2
SNAPSHOT_URL = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/snapshot/{DATASET_VERSION}"
)
FILES_URL = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?"
    f"folder_id=root&version={DATASET_VERSION}"
)
ZIP_NAME = "ML-CFD-Wavy-Channel-Surrogate.zip"

FEATURE_ALIASES = {
    "Re": ["re", "reynolds", "reynolds_number", "reynoldsnumber"],
    "Pr": ["pr", "prandtl", "prandtl_number", "prandtlnumber"],
    "Da": ["da", "darcy", "darcy_number", "darcynumber"],
    "porosity": ["porosity", "epsilon", "eps", "epsi", "phi", "por"],
    "thickness": [
        "porous_slab_thickness",
        "slab_thickness",
        "porous_layer_thickness",
        "thickness",
        "t",
        "tp",
        "h_p",
        "hp",
        "hp_mm",
    ],
    "amplitude": ["wave_amplitude", "amplitude", "amp", "a", "a_mm"],
    "wavelength": ["wavelength", "wave_length", "lambda", "lambda_wave", "wl", "lw", "l", "lw_mm"],
}

TARGET_ALIASES = {
    "Nuavg": [
        "nuavg",
        "nu_avg",
        "nu_average",
        "average_nusselt_number",
        "nusselt",
        "nusselt_number",
    ],
    "DelP_Pa": [
        "delp_pa",
        "delta_p_pa",
        "pressure_drop_pa",
        "pressure_drop",
        "deltap",
        "delp",
        "dp",
    ],
}


def _request_json(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
            "User-Agent": "rcsr-experiment/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(force: bool = False) -> Path:
    ensure_dirs()
    snapshot = _request_json(SNAPSHOT_URL)
    files = _request_json(FILES_URL)

    (RAW_DATA_DIR / "mendeley_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    (RAW_DATA_DIR / "mendeley_files.json").write_text(
        json.dumps(files, indent=2), encoding="utf-8"
    )

    archive_entry = next(
        (item for item in files if item.get("filename", "").lower().endswith(".zip")),
        None,
    )
    if archive_entry is None:
        raise RuntimeError("No ZIP archive found in the public Mendeley file listing.")

    details = archive_entry["content_details"]
    archive_path = RAW_DATA_DIR / archive_entry["filename"]
    expected_size = int(details.get("size", 0))
    expected_hash = details.get("sha256_hash")

    if archive_path.exists() and not force:
        size_ok = expected_size == 0 or archive_path.stat().st_size == expected_size
        hash_ok = not expected_hash or _sha256(archive_path) == expected_hash
        if size_ok and hash_ok:
            print(f"Archive already present: {archive_path}")
        else:
            print("Existing archive failed size/hash check; downloading again.")
            force = True

    if force or not archive_path.exists():
        url = details["download_url"]
        request = urllib.request.Request(url, headers={"User-Agent": "rcsr-experiment/1.0"})
        print(f"Downloading {archive_entry['filename']} ({expected_size / 1e6:.1f} MB)")
        with urllib.request.urlopen(request, timeout=120) as response, archive_path.open(
            "wb"
        ) as out:
            downloaded = 0
            next_report = 50 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  downloaded {downloaded / 1e6:.1f} MB")
                    next_report += 50 * 1024 * 1024

        if expected_size and archive_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Downloaded archive size mismatch: {archive_path.stat().st_size} "
                f"!= {expected_size}"
            )
        if expected_hash:
            observed_hash = _sha256(archive_path)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Downloaded archive SHA256 mismatch: {observed_hash} != {expected_hash}"
                )

    extract_dir = RAW_DATA_DIR / "ML-CFD-Wavy-Channel-Surrogate"
    marker = extract_dir / ".extract_complete"
    if force or not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {extract_dir}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")

    return discover_dataset_csv()


def discover_dataset_csv() -> Path:
    candidates = list(RAW_DATA_DIR.rglob("ML_dataset_longform.csv"))
    if candidates:
        return candidates[0]
    csvs = list(RAW_DATA_DIR.rglob("*.csv"))
    scored = []
    for path in csvs:
        name = path.name.lower()
        score = 0
        if "longform" in name:
            score += 5
        if "dataset" in name:
            score += 3
        if "ml" in name:
            score += 1
        scored.append((score, path))
    if scored:
        return sorted(scored, reverse=True)[0][1]
    raise FileNotFoundError("Could not locate the processed dataset CSV.")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _resolve_alias(columns: Iterable[str], aliases: List[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    alias_norms = {_norm(alias) for alias in aliases}
    for alias in alias_norms:
        if alias in normalized:
            return normalized[alias]
    for normed, original in normalized.items():
        if any(alias in normed for alias in alias_norms if len(alias) > 2):
            return original
    return None


def canonicalize_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    for canonical, aliases in {**FEATURE_ALIASES, **TARGET_ALIASES}.items():
        found = _resolve_alias(df.columns, aliases)
        if found is not None:
            mapping[canonical] = found

    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    target_like = set(mapping.values())
    remaining = [col for col in numeric_cols if col not in target_like]

    for target, tokens in TARGET_ALIASES.items():
        if target not in mapping:
            found = next(
                (
                    col
                    for col in numeric_cols
                    if any(token in _norm(col) for token in tokens)
                ),
                None,
            )
            if found is not None:
                mapping[target] = found

    feature_names = list(FEATURE_ALIASES)
    missing_features = [name for name in feature_names if name not in mapping]
    if missing_features and len(remaining) >= len(feature_names):
        for name, col in zip(missing_features, remaining):
            if col not in mapping.values():
                mapping[name] = col

    missing_targets = [name for name in TARGET_ALIASES if name not in mapping]
    if missing_targets:
        candidates = [col for col in numeric_cols if col not in mapping.values()]
        if len(candidates) >= len(missing_targets):
            for name, col in zip(missing_targets, candidates[-len(missing_targets) :]):
                mapping[name] = col

    required = feature_names + list(TARGET_ALIASES)
    missing = [col for col in required if col not in mapping]
    if missing:
        raise ValueError(
            f"Could not resolve required columns {missing}. "
            f"Available columns: {list(df.columns)}. Partial mapping: {mapping}"
        )

    out = pd.DataFrame({canonical: df[original] for canonical, original in mapping.items()})
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    return out[required], mapping


def load_prepared_dataset() -> Tuple[pd.DataFrame, Dict[str, str], Path]:
    ensure_dirs()
    csv_path = discover_dataset_csv()
    df = pd.read_csv(csv_path)
    canonical, mapping = canonicalize_dataset(df)
    processed_path = PROCESSED_DATA_DIR / "canonical_wavy_channel_dataset.csv"
    canonical.to_csv(processed_path, index=False)
    (PROCESSED_DATA_DIR / "column_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    return canonical, mapping, csv_path


def dataset_profile(df: pd.DataFrame) -> Dict[str, object]:
    features = list(FEATURE_ALIASES)
    targets = list(TARGET_ALIASES)
    profile = {
        "n_rows": int(len(df)),
        "features": features,
        "targets": targets,
        "feature_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0)),
            }
            for col in features
        },
        "target_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0)),
            }
            for col in targets
        },
    }
    geom = df[["thickness", "amplitude", "wavelength"]].drop_duplicates()
    profile["n_geometry_combinations"] = int(len(geom))
    return profile
