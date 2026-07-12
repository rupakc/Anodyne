"""anodyne-perturbation: controlled corruption (noise/drift/outliers/bias/edge-case)
applied to a stored dataset artifact to produce a derived `DatasetVersion`."""

from anodyne_perturbation.perturbator import RegistryPerturbator
from anodyne_perturbation.registry import (
    get_perturbation_handler,
    register_perturbation,
    registered_perturbation_modalities,
)

__all__ = [
    "RegistryPerturbator",
    "get_perturbation_handler",
    "register_perturbation",
    "registered_perturbation_modalities",
]
