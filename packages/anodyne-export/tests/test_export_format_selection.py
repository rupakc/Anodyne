from anodyne_export.exporter import LARGE_DATASET_ROW_THRESHOLD, resolve_format


def test_threshold_is_the_named_constant() -> None:
    assert LARGE_DATASET_ROW_THRESHOLD == 500_000


def test_small_dataset_defaults_to_csv() -> None:
    assert resolve_format(10, None) == "csv"


def test_boundary_row_count_still_defaults_to_csv() -> None:
    # Threshold rule is "> 500_000", not ">=" -- exactly at the boundary stays CSV.
    assert resolve_format(500_000, None) == "csv"


def test_over_threshold_defaults_to_parquet() -> None:
    assert resolve_format(500_001, None) == "parquet"


def test_large_dataset_defaults_to_parquet() -> None:
    assert resolve_format(2_000_000, None) == "parquet"


def test_explicit_format_always_wins_regardless_of_row_count() -> None:
    assert resolve_format(2_000_000, "json") == "json"
    assert resolve_format(10, "arrow") == "arrow"
