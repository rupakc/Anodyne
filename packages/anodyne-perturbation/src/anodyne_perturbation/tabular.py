"""Tabular-modality perturbations: the five families over numeric / categorical
/ string arrow columns.

Determinism follows the repo idiom: an independent
`np.random.default_rng([seed, family_ordinal, field_index])` stream per column,
so the same `(seed, spec)` always yields byte-identical output and different
seeds differ. Column dtypes are preserved (int stays int via rounding) so the
derived artifact keeps the parent's schema.
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


def _family_ord(family: PerturbationFamily) -> int:
    return list(PerturbationFamily).index(family)


def _is_numeric(t: pa.DataType) -> bool:
    return bool(pa.types.is_integer(t) or pa.types.is_floating(t))


def _selected(table: pa.Table, spec: PerturbationSpec, predicate: Any) -> list[str]:
    targets = set(spec.target_fields)
    out = []
    for i, name in enumerate(table.column_names):
        t = table.schema.field(i).type
        if not predicate(t):
            continue
        if targets and name not in targets:
            continue
        out.append(name)
    return out


def _rebuild(
    table: pa.Table, cols: dict[str, list[Any]], types: dict[str, pa.DataType]
) -> pa.Table:
    arrays = []
    for name in table.column_names:
        t = types[name]
        try:
            if pa.types.is_integer(t):
                vals = [None if v is None else int(round(float(v))) for v in cols[name]]
                arrays.append(pa.array(vals, type=t))
            else:
                arrays.append(pa.array(cols[name], type=t))
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, OverflowError):
            # A family widened the value beyond the original type (e.g. format
            # violations); fall back to an inferred type rather than crashing.
            arrays.append(pa.array(cols[name]))
    return pa.table(dict(zip(table.column_names, arrays, strict=True)))


def perturb_tabular(spec: PerturbationSpec, table: pa.Table, seed: int) -> pa.Table:
    """Apply `spec` to `table`, returning a corrupted copy with the same schema."""
    if table.num_rows == 0:
        return table
    cols: dict[str, list[Any]] = {n: table.column(n).to_pylist() for n in table.column_names}
    types = {n: table.schema.field(n).type for n in table.column_names}
    n = table.num_rows
    family = spec.family
    fam_ord = _family_ord(family)

    if family is PerturbationFamily.NOISE:
        _noise(spec, cols, types, table, seed, fam_ord)
    elif family is PerturbationFamily.DRIFT:
        _drift(spec, cols, types, table, seed, fam_ord, n)
    elif family is PerturbationFamily.OUTLIERS:
        _outliers(spec, cols, types, table, seed, fam_ord)
    elif family is PerturbationFamily.BIAS:
        order = _bias(spec, cols, n, seed)
        cols = {name: [cols[name][i] for i in order] for name in cols}
    elif family is PerturbationFamily.EDGE_CASE:
        _edge_case(spec, cols, types, table, seed, fam_ord)

    return _rebuild(table, cols, types)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def _noise(spec, cols, types, table, seed, fam_ord) -> None:  # type: ignore[no-untyped-def]
    p = parse_params(spec)
    assert isinstance(p, NoiseParams)
    for fi, name in enumerate(_selected(table, spec, _is_numeric)):
        rng = np.random.default_rng([seed, fam_ord, fi])
        arr = np.array([float(v) if v is not None else np.nan for v in cols[name]])
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        std = float(np.std(finite)) or 1.0
        if p.kind == "uniform":
            span = (float(np.max(finite)) - float(np.min(finite))) or 1.0
            noise = rng.uniform(-spec.intensity * span, spec.intensity * span, arr.shape)
        else:
            noise = rng.normal(0.0, spec.intensity * std, arr.shape)
        cols[name] = [None if v is None else (v + noise[i]) for i, v in enumerate(cols[name])]
    # Categorical (string) columns: random swap to another observed value.
    for fi, name in enumerate(_selected(table, spec, pa.types.is_string)):
        rng = np.random.default_rng([seed, fam_ord, 1000 + fi])
        observed = sorted({v for v in cols[name] if v is not None})
        if len(observed) < 2:
            continue
        new = []
        for v in cols[name]:
            if v is not None and rng.random() < spec.intensity:
                choices = [c for c in observed if c != v]
                new.append(choices[int(rng.integers(0, len(choices)))])
            else:
                new.append(v)
        cols[name] = new


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def _drift(spec, cols, types, table, seed, fam_ord, n) -> None:  # type: ignore[no-untyped-def]
    d = parse_params(spec)
    assert isinstance(d, DriftParams)
    if d.kind == "concept":
        for name in _selected(table, spec, pa.types.is_string):
            if d.relabel:
                cols[name] = [d.relabel.get(str(v), v) for v in cols[name]]
            else:  # rotate categories deterministically
                observed = sorted({v for v in cols[name] if v is not None})
                if len(observed) >= 2:
                    k = len(observed)
                    rot = {observed[i]: observed[(i + 1) % k] for i in range(k)}
                    cols[name] = [rot.get(v, v) for v in cols[name]]
        return
    for name in _selected(table, spec, _is_numeric):
        arr = np.array([float(v) if v is not None else np.nan for v in cols[name]])
        finite = arr[np.isfinite(arr)]
        std = float(np.std(finite)) if finite.size else 1.0
        if d.kind == "temporal":
            trend = spec.intensity * d.slope * std * (np.arange(n) / max(n - 1, 1))
            cols[name] = [None if v is None else v + trend[i] for i, v in enumerate(cols[name])]
        else:  # covariate: scale + shift
            shift = d.shift if d.shift is not None else spec.intensity * std
            cols[name] = [None if v is None else v * d.scale + shift for v in cols[name]]


# --------------------------------------------------------------------------- #
# Outliers / anomalies
# --------------------------------------------------------------------------- #
def _outliers(spec, cols, types, table, seed, fam_ord) -> None:  # type: ignore[no-untyped-def]
    o = parse_params(spec)
    assert isinstance(o, OutlierParams)
    if o.kind == "contextual":
        for fi, name in enumerate(_selected(table, spec, pa.types.is_string)):
            rng = np.random.default_rng([seed, fam_ord, fi])
            cols[name] = [
                o.rare_value if (v is not None and rng.random() < spec.intensity) else v
                for v in cols[name]
            ]
        return
    for fi, name in enumerate(_selected(table, spec, _is_numeric)):
        rng = np.random.default_rng([seed, fam_ord, fi])
        arr = np.array([float(v) if v is not None else np.nan for v in cols[name]])
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        mean, std = float(np.mean(finite)), float(np.std(finite)) or 1.0
        new = []
        for v in cols[name]:
            if v is not None and rng.random() < spec.intensity:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                new.append(mean + sign * o.magnitude * std)
            else:
                new.append(v)
        cols[name] = new


# --------------------------------------------------------------------------- #
# Bias (row resampling) -- shared with the text handler.
# --------------------------------------------------------------------------- #
def bias_reindex(
    params: BiasParams,
    spec: PerturbationSpec,
    cols: dict[str, list[Any]],
    n: int,
    seed: int,
) -> list[int]:
    """Return a length-`n` row index order that over-represents a target value.

    Picks the class/demographic field (`params.field`, else `target_fields[0]`,
    else the first column), a target value (`params.target_value`, else the most
    frequent value), and resamples with replacement so ~`target_ratio` of rows
    carry the target value. Row count is preserved.
    """
    rng = np.random.default_rng([seed, _family_ord(PerturbationFamily.BIAS)])
    field = params.field or (spec.target_fields[0] if spec.target_fields else None)
    if field is None or field not in cols:
        field = next(iter(cols))
    values = cols[field]
    distinct = [v for v in dict.fromkeys(values)]
    if len(distinct) < 2:
        return list(range(n))
    if params.target_value is not None:
        target = params.target_value
        pos_idx = [i for i, v in enumerate(values) if str(v) == str(target)]
    else:
        counts = {v: values.count(v) for v in distinct}
        target = max(counts, key=lambda k: counts[k])
        pos_idx = [i for i, v in enumerate(values) if v == target]
    neg_idx = [i for i in range(n) if i not in set(pos_idx)]
    if not pos_idx or not neg_idx:
        return list(range(n))
    n_pos = int(round(params.target_ratio * n))
    n_neg = n - n_pos
    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=True).tolist()
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=True).tolist()
    order = [int(i) for i in chosen_pos + chosen_neg]
    rng.shuffle(order)
    return order


def _bias(spec, cols, n, seed) -> list[int]:  # type: ignore[no-untyped-def]
    b = parse_params(spec)
    assert isinstance(b, BiasParams)
    return bias_reindex(b, spec, cols, n, seed)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def _edge_case(spec, cols, types, table, seed, fam_ord) -> None:  # type: ignore[no-untyped-def]
    e = parse_params(spec)
    assert isinstance(e, EdgeCaseParams)
    for fi, name in enumerate(table.column_names):
        if spec.target_fields and name not in spec.target_fields:
            continue
        t = types[name]
        rng = np.random.default_rng([seed, fam_ord, fi])
        numeric = _is_numeric(t)
        is_str = pa.types.is_string(t)
        if not numeric and not is_str:
            continue
        bound_lo = bound_hi = None
        if numeric:
            finite = [float(v) for v in cols[name] if v is not None]
            if finite:
                bound_lo, bound_hi = min(finite), max(finite)
        new = []
        for v in cols[name]:
            if rng.random() >= spec.intensity:
                new.append(v)
                continue
            if e.kind == "nulls":
                new.append(None)
            elif e.kind == "boundary" and numeric and bound_lo is not None:
                new.append(bound_hi if rng.random() < 0.5 else bound_lo)
            elif e.kind == "boundary" and is_str:
                new.append("")
            elif e.kind == "format":
                new.append(0 if numeric else "   ")
            else:
                new.append(None)
        cols[name] = new
