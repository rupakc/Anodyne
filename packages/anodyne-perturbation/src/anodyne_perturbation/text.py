"""Text-modality perturbations + reusable per-string corruption ops.

All ops are pure and seeded (`np.random.default_rng`) -- no network, no models
-- so the whole family is fast and offline-testable. The string ops
(`char_typos`/`word_typos`/`mask_words`) are also imported by the tabular noise
family for its string columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec

from anodyne_perturbation.params import (
    BiasParams,
    DriftParams,
    EdgeCaseParams,
    NoiseParams,
    OutlierParams,
    parse_params,
)

_MASK = "[MASK]"
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def char_typos(s: str, rng: np.random.Generator, intensity: float) -> str:
    """Corrupt roughly `intensity` of characters (swap-adjacent/delete/insert)."""
    if not s:
        return s
    chars = list(s)
    out: list[str] = []
    for ch in chars:
        if rng.random() < intensity:
            op = int(rng.integers(0, 3))
            if op == 0:  # delete
                continue
            if op == 1:  # insert a random letter before
                out.append(_ALPHABET[int(rng.integers(0, len(_ALPHABET)))])
                out.append(ch)
            else:  # substitute
                out.append(_ALPHABET[int(rng.integers(0, len(_ALPHABET)))])
        else:
            out.append(ch)
    return "".join(out)


def word_typos(s: str, rng: np.random.Generator, intensity: float) -> str:
    """Apply `char_typos` to roughly `intensity` of the whitespace-split words."""
    words = s.split(" ")
    return " ".join(char_typos(w, rng, 0.5) if rng.random() < intensity else w for w in words)


def mask_words(s: str, rng: np.random.Generator, intensity: float) -> str:
    """Replace roughly `intensity` of words with a `[MASK]` token."""
    words = s.split(" ")
    return " ".join(_MASK if rng.random() < intensity else w for w in words)


def _string_columns(table: pa.Table, spec: PerturbationSpec) -> list[str]:
    targets = set(spec.target_fields)
    names = []
    for i, name in enumerate(table.column_names):
        if not pa.types.is_string(table.schema.field(i).type):
            continue
        if targets and name not in targets:
            continue
        names.append(name)
    return names


def perturb_text(spec: PerturbationSpec, table: pa.Table, seed: int) -> pa.Table:
    """Perturb the string columns of a text artifact per `spec.family`."""
    if table.num_rows == 0:
        return table
    family = spec.family
    cols: dict[str, list[Any]] = {n: table.column(n).to_pylist() for n in table.column_names}
    string_cols = _string_columns(table, spec)
    n = table.num_rows

    if family is PerturbationFamily.BIAS:
        order = _bias_reindex(spec, cols, n, seed)
        cols = {name: [cols[name][i] for i in order] for name in cols}
    elif family is PerturbationFamily.DRIFT:
        d = parse_params(spec)
        assert isinstance(d, DriftParams)
        for fi, name in enumerate(string_cols):
            rng = np.random.default_rng([seed, int(family_ord(family)), fi])
            cols[name] = _text_drift(cols[name], d, spec, rng, n)
    else:
        for fi, name in enumerate(string_cols):
            rng = np.random.default_rng([seed, int(family_ord(family)), fi])
            cols[name] = [
                _apply_cell(family, spec, str(v) if v is not None else "", rng) for v in cols[name]
            ]

    arrays = [pa.array(cols[name]) for name in table.column_names]
    return pa.table(dict(zip(table.column_names, arrays, strict=True)))


def family_ord(family: PerturbationFamily) -> int:
    return list(PerturbationFamily).index(family)


def _apply_cell(
    family: PerturbationFamily,
    spec: PerturbationSpec,
    value: str,
    rng: np.random.Generator,
) -> str:
    if family is PerturbationFamily.NOISE:
        p = parse_params(spec)
        assert isinstance(p, NoiseParams)
        if p.text_op == "word":
            return word_typos(value, rng, spec.intensity)
        if p.text_op == "mask":
            return mask_words(value, rng, spec.intensity)
        return char_typos(value, rng, spec.intensity)
    if family is PerturbationFamily.OUTLIERS:
        o = parse_params(spec)
        assert isinstance(o, OutlierParams)
        if rng.random() < spec.intensity:
            return o.rare_value * max(1, int(o.magnitude))
        return value
    if family is PerturbationFamily.EDGE_CASE:
        e = parse_params(spec)
        assert isinstance(e, EdgeCaseParams)
        if rng.random() < max(spec.intensity, 0.0):
            return "" if e.kind != "format" else "   "
        return value
    return value


def _text_drift(
    values: list[Any],
    params: DriftParams,
    spec: PerturbationSpec,
    rng: np.random.Generator,
    n: int,
) -> list[Any]:
    if params.kind == "concept" and params.relabel:
        return [params.relabel.get(str(v), v) for v in values]
    # temporal / covariate for text: progressively heavier char typos over row order.
    out = []
    for i, v in enumerate(values):
        rate = spec.intensity * (i / max(n - 1, 1)) * params.slope
        out.append(char_typos(str(v) if v is not None else "", rng, min(rate, 1.0)))
    return out


def _bias_reindex(
    spec: PerturbationSpec, cols: dict[str, list[Any]], n: int, seed: int
) -> list[int]:
    from anodyne_perturbation.tabular import bias_reindex

    b = parse_params(spec)
    assert isinstance(b, BiasParams)
    return bias_reindex(b, spec, cols, n, seed)
