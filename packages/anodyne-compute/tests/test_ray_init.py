"""Unit tests for `anodyne_compute.ray_init` using a mocked `ray` module.

No live Ray cluster/process is started here (that's covered by the
`integration`-marked tests in `test_ray_tasks.py`); these just verify the
address-selection and idempotency logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from anodyne_compute.ray_tasks import ray_init


def test_ray_init_is_noop_when_already_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    monkeypatch.setattr("anodyne_compute.ray_tasks.ray", mock_ray)

    ray_init("ray://host:10001")

    mock_ray.init.assert_not_called()


def test_ray_init_connects_to_given_address(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = False
    monkeypatch.setattr("anodyne_compute.ray_tasks.ray", mock_ray)

    ray_init("ray://host:10001")

    mock_ray.init.assert_called_once_with(address="ray://host:10001")


def test_ray_init_falls_back_to_local_when_address_falsy(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = False
    monkeypatch.setattr("anodyne_compute.ray_tasks.ray", mock_ray)

    ray_init("")

    mock_ray.init.assert_called_once_with()
