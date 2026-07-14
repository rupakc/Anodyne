from __future__ import annotations

import json
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from anodyne_evaluation.loader import load_manifest


def test_loads_dict_form_with_items() -> None:
    """Test loading manifest in dict form with items list."""
    manifest = {
        "items": [
            {
                "item_index": 0,
                "object_key": "image1.png",
                "prompt": "describe this image",
                "label": "cat",
                "mime_type": "image/png",
            },
            {
                "item_index": 1,
                "object_key": "image2.png",
                "prompt": "identify the object",
                "label": "dog",
                "mime_type": "image/png",
            },
        ]
    }
    data = json.dumps(manifest).encode()
    result = load_manifest(data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == [
        "item_index",
        "object_key",
        "prompt",
        "label",
        "mime_type",
    ]
    assert result.iloc[0]["label"] == "cat"
    assert result.iloc[1]["label"] == "dog"


def test_loads_bare_list_form() -> None:
    """Test loading manifest as a bare list."""
    manifest = [
        {
            "item_index": 0,
            "object_key": "audio1.mp3",
            "prompt": "transcribe",
            "label": "speech",
            "mime_type": "audio/mp3",
        },
        {
            "item_index": 1,
            "object_key": "audio2.mp3",
            "prompt": "transcribe",
            "label": "music",
            "mime_type": "audio/mp3",
        },
    ]
    data = json.dumps(manifest).encode()
    result = load_manifest(data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "item_index" in result.columns
    assert result.iloc[0]["label"] == "speech"


def test_empty_items_returns_empty_dataframe() -> None:
    """Test that empty items returns an empty DataFrame with no error."""
    manifest: dict[str, Any] = {"items": []}
    data = json.dumps(manifest).encode()
    result = load_manifest(data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_bare_empty_list_returns_empty_dataframe() -> None:
    """Test that a bare empty list returns an empty DataFrame."""
    manifest: list[Any] = []
    data = json.dumps(manifest).encode()
    result = load_manifest(data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
