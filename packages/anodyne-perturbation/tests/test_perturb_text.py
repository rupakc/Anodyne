import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.text import char_typos, mask_words, perturb_text


def _table() -> pa.Table:
    return pa.table(
        {
            "review": pa.array(["the quick brown fox jumps over the lazy dog"] * 50),
            "label": pa.array(["pos", "neg"] * 25),
        }
    )


def test_char_typos_change_string_deterministically() -> None:
    rng1 = np.random.default_rng([1, 2])
    rng2 = np.random.default_rng([1, 2])
    s = "hello world this is a test"
    a = char_typos(s, rng1, 0.5)
    b = char_typos(s, rng2, 0.5)
    assert a == b
    assert a != s


def test_mask_words_inserts_mask_token() -> None:
    rng = np.random.default_rng([9])
    out = mask_words("one two three four five six", rng, 0.9)
    assert "[MASK]" in out


def test_text_noise_perturbs_string_column() -> None:
    table = _table()
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.4)
    out = perturb_text(spec, table, seed=1)
    assert out.column("review").to_pylist()[0] != table.column("review").to_pylist()[0]
    assert out.num_rows == table.num_rows


def test_text_concept_drift_relabels() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.DRIFT,
        target_fields=["label"],
        params={"kind": "concept", "relabel": {"pos": "POSITIVE"}},
    )
    out = perturb_text(spec, table, seed=1)
    assert "POSITIVE" in out.column("label").to_pylist()


def test_text_edge_case_empties_strings() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.EDGE_CASE, intensity=1.0, target_fields=["review"]
    )
    out = perturb_text(spec, table, seed=1)
    assert out.column("review").to_pylist()[0] == ""
