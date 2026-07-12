from __future__ import annotations

from uuid import uuid4

from anodyne_dataset.directives import parse_directives
from anodyne_dataset.models import Modality
from anodyne_templates.catalog import build_dataset_spec, get_template, list_templates

_REQUIRED_KEYS = {"customers", "transactions", "support_tickets", "sensor_readings", "users_churn"}


def test_catalog_covers_required_use_cases() -> None:
    templates = list_templates()
    keys = [t.key for t in templates]

    assert len(keys) == len(set(keys))  # unique keys
    assert _REQUIRED_KEYS <= set(keys)


def test_every_template_has_fields_and_positive_default_rows() -> None:
    for template in list_templates():
        assert len(template.fields) >= 1
        assert template.default_target_rows > 0
        assert template.modality is Modality.TABULAR


def test_get_template_hit_and_miss() -> None:
    assert get_template("customers") is not None
    assert get_template("customers").key == "customers"  # type: ignore[union-attr]
    assert get_template("does-not-exist") is None


def test_build_dataset_spec_uses_template_defaults() -> None:
    template = get_template("customers")
    assert template is not None
    tenant_id = uuid4()

    spec = build_dataset_spec(template, tenant_id=tenant_id)

    assert spec.tenant_id == tenant_id
    assert spec.source == "template"
    assert spec.status == "draft"
    assert spec.modality is Modality.TABULAR
    assert spec.fields == template.fields
    assert spec.target_rows == template.default_target_rows
    assert spec.directives == template.default_directives


def test_build_dataset_spec_overrides_win_over_defaults() -> None:
    template = get_template("customers")
    assert template is not None

    spec = build_dataset_spec(
        template,
        tenant_id=uuid4(),
        name="My customers",
        target_rows=42,
        directives={"directives": [{"kind": "bias", "field": "plan", "value": "pro", "rate": 0.5}]},
    )

    assert spec.name == "My customers"
    assert spec.target_rows == 42
    assert spec.directives["directives"][0]["field"] == "plan"  # type: ignore[index]


def test_users_churn_template_ships_a_default_rare_event_directive() -> None:
    template = get_template("users_churn")
    assert template is not None
    directives = parse_directives(template.default_directives)
    assert any(d.name == "rare_event" for d in directives)
