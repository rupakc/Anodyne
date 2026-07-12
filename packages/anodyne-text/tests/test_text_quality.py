from anodyne_text.quality import Deduplicator, passes_quality


def test_empty_primary_fails() -> None:
    assert not passes_quality({"text": "", "label": "a"}, "text")


def test_whitespace_only_primary_fails() -> None:
    assert not passes_quality({"text": "   \n\t  ", "label": "a"}, "text")


def test_missing_primary_fails() -> None:
    assert not passes_quality({"label": "a"}, "text")


def test_too_short_fails() -> None:
    assert not passes_quality({"text": "hi"}, "text", min_length=5)


def test_too_long_fails() -> None:
    assert not passes_quality({"text": "x" * 100}, "text", max_length=50)


def test_within_bounds_passes() -> None:
    assert passes_quality({"text": "a reasonable sentence"}, "text", min_length=5, max_length=100)


def test_default_bounds_accept_typical_text() -> None:
    assert passes_quality({"text": "a normal short row of text"}, "text")


def test_deduplicator_flags_exact_repeat() -> None:
    dedup = Deduplicator()
    row = {"text": "same content"}
    assert not dedup.is_duplicate(row, "text")
    assert dedup.is_duplicate(row, "text")


def test_deduplicator_normalizes_whitespace() -> None:
    dedup = Deduplicator()
    assert not dedup.is_duplicate({"text": "hello world"}, "text")
    assert dedup.is_duplicate({"text": "  hello world  "}, "text")


def test_deduplicator_distinct_values_not_flagged() -> None:
    dedup = Deduplicator()
    assert not dedup.is_duplicate({"text": "a"}, "text")
    assert not dedup.is_duplicate({"text": "b"}, "text")


def test_deduplicator_is_per_instance() -> None:
    row = {"text": "same content"}
    dedup1 = Deduplicator()
    dedup1.is_duplicate(row, "text")
    dedup2 = Deduplicator()
    assert not dedup2.is_duplicate(row, "text")
