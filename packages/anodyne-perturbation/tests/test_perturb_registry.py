import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation import (
    RegistryPerturbator,
    get_perturbation_handler,
    registered_perturbation_modalities,
)


def test_all_modalities_registered() -> None:
    assert registered_perturbation_modalities() == [
        "audio",
        "image",
        "tabular",
        "text",
        "video",
    ]


def test_unknown_modality_falls_back_to_tabular() -> None:
    assert get_perturbation_handler("does-not-exist") is get_perturbation_handler("tabular")


def test_media_modalities_are_a_seam_not_fake_logic() -> None:
    handler = get_perturbation_handler("image")
    spec = PerturbationSpec(family=PerturbationFamily.NOISE)
    with pytest.raises(NotImplementedError):
        handler.perturb(spec, pa.table({"x": [1, 2, 3]}), 1)


def test_registry_perturbator_dispatches_by_modality() -> None:
    perturbator = RegistryPerturbator()
    table = pa.table({"x": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())})
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.5)
    out = perturbator.perturb(spec, table, "tabular", 1)
    assert out.num_rows == 4
    assert out.column("x").to_pylist() != table.column("x").to_pylist()
