"""Starter template catalog: ready-made `DatasetSpec` blueprints for common use-cases.

A user picks a template and customizes it, instead of writing a from-scratch description. `GET
/templates` (gateway) lists `list_templates()`; `POST /datasets/from-template` builds a
`DatasetSpec` via `build_dataset_spec` and persists it with `source="template"`.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_dataset.models import DatasetSpec, FieldSpec, SemanticType

from anodyne_templates.models import DatasetTemplate

TEMPLATES: list[DatasetTemplate] = [
    DatasetTemplate(
        key="customers",
        name="Customers",
        description="A customer roster: identity, signup, plan tier, and location.",
        category="crm",
        fields=[
            FieldSpec(name="full_name", semantic_type=SemanticType.NAME),
            FieldSpec(name="email", semantic_type=SemanticType.EMAIL),
            FieldSpec(name="signup_date", semantic_type=SemanticType.DATETIME),
            FieldSpec(
                name="plan",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["free", "pro", "enterprise"]},
            ),
            FieldSpec(name="country", semantic_type=SemanticType.ADDRESS),
        ],
        default_target_rows=1_000,
    ),
    DatasetTemplate(
        key="transactions",
        name="Transactions",
        description="Payment transactions with amount, currency, and a fraud flag.",
        category="finance",
        fields=[
            FieldSpec(
                name="amount",
                semantic_type=SemanticType.FLOAT,
                constraints={"min": 1.0, "max": 5_000.0},
            ),
            FieldSpec(
                name="currency",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["USD", "EUR", "GBP"]},
            ),
            FieldSpec(name="timestamp", semantic_type=SemanticType.DATETIME),
            FieldSpec(name="is_fraud", semantic_type=SemanticType.BOOLEAN),
        ],
        default_target_rows=5_000,
    ),
    DatasetTemplate(
        key="support_tickets",
        name="Support tickets",
        description="Customer support tickets with priority, status, and resolution.",
        category="support",
        fields=[
            FieldSpec(name="subject", semantic_type=SemanticType.TEXT),
            FieldSpec(
                name="priority",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["low", "medium", "high", "urgent"]},
            ),
            FieldSpec(
                name="status",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["open", "pending", "resolved", "closed"]},
            ),
            FieldSpec(name="created_at", semantic_type=SemanticType.DATETIME),
            FieldSpec(name="resolved", semantic_type=SemanticType.BOOLEAN),
        ],
        default_target_rows=2_000,
    ),
    DatasetTemplate(
        key="sensor_readings",
        name="Sensor readings",
        description="IoT sensor telemetry with temperature, humidity, and anomaly flags.",
        category="iot",
        fields=[
            FieldSpec(
                name="sensor_id",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": [f"sensor-{i:03d}" for i in range(1, 21)]},
            ),
            FieldSpec(
                name="temperature",
                semantic_type=SemanticType.FLOAT,
                constraints={"min": -10.0, "max": 45.0},
            ),
            FieldSpec(
                name="humidity",
                semantic_type=SemanticType.FLOAT,
                constraints={"min": 0.0, "max": 100.0},
            ),
            FieldSpec(name="reading_at", semantic_type=SemanticType.DATETIME),
            FieldSpec(name="anomaly", semantic_type=SemanticType.BOOLEAN),
        ],
        default_target_rows=10_000,
    ),
    DatasetTemplate(
        key="users_churn",
        name="Users + churn label",
        description="User accounts with usage stats and a churn label, pre-biased rare.",
        category="ml-labels",
        fields=[
            FieldSpec(
                name="age", semantic_type=SemanticType.INTEGER, constraints={"min": 18, "max": 90}
            ),
            FieldSpec(
                name="tenure_months",
                semantic_type=SemanticType.INTEGER,
                constraints={"min": 0, "max": 120},
            ),
            FieldSpec(
                name="monthly_spend",
                semantic_type=SemanticType.FLOAT,
                constraints={"min": 0.0, "max": 500.0},
            ),
            FieldSpec(
                name="plan",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["free", "pro", "enterprise"]},
            ),
            FieldSpec(name="churned", semantic_type=SemanticType.BOOLEAN),
        ],
        default_target_rows=3_000,
        # Demonstrates directive-driven class imbalance out of the box: churn is
        # naturally a rare positive label, so nudge it via a "rare_event" use_case
        # directive rather than shipping a naive ~50/50 boolean.
        default_directives={
            "directives": [
                {"kind": "use_case", "name": "rare_event", "field": "churned", "value": True}
            ]
        },
    ),
]

_BY_KEY: dict[str, DatasetTemplate] = {t.key: t for t in TEMPLATES}


def list_templates() -> list[DatasetTemplate]:
    return list(TEMPLATES)


def get_template(key: str) -> DatasetTemplate | None:
    return _BY_KEY.get(key)


def build_dataset_spec(
    template: DatasetTemplate,
    *,
    tenant_id: UUID,
    name: str | None = None,
    target_rows: int | None = None,
    directives: dict[str, object] | None = None,
) -> DatasetSpec:
    """Build a `DatasetSpec` (`source="template"`) from a catalog template.

    Explicit `name`/`target_rows`/`directives` override the template's defaults; omitted
    arguments fall back to the template.
    """
    return DatasetSpec(
        id=uuid4(),
        tenant_id=tenant_id,
        name=name if name is not None else template.name,
        description=template.description,
        modality=template.modality,
        source="template",
        fields=list(template.fields),
        target_rows=target_rows if target_rows is not None else template.default_target_rows,
        directives=directives if directives is not None else dict(template.default_directives),
    )


__all__ = ["TEMPLATES", "list_templates", "get_template", "build_dataset_spec"]
