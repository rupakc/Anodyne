"""Deterministic, realistic values for PII-like fields (name/email/address/text).

These columns are never sampled from the fitted statistical/deep model -- copying a real
sample's names/emails/addresses into "synthetic" output would leak the source data. Instead
they're generated with Faker (default) or Mimesis (`constraints={"provider": "mimesis"}`),
seeded so the same `(field, count, rng_seed)` always produces the same values.
"""

from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import FieldSpec, SemanticType
from faker import Faker


def faker_column(field: FieldSpec, count: int, rng_seed: int) -> pa.Array:
    """Generate `count` realistic values for `field` (NAME/EMAIL/ADDRESS/TEXT), seeded."""
    provider = field.constraints.get("provider", "faker")
    if provider == "mimesis":
        return _mimesis_column(field, count, rng_seed)
    return _faker_column(field, count, rng_seed)


def _faker_column(field: FieldSpec, count: int, rng_seed: int) -> pa.Array:
    locale = field.constraints.get("faker_locale")
    fake = Faker(str(locale)) if locale else Faker()
    Faker.seed(rng_seed)
    st = field.semantic_type
    if st is SemanticType.NAME:
        return pa.array([fake.name() for _ in range(count)])
    if st is SemanticType.EMAIL:
        return pa.array([fake.email() for _ in range(count)])
    if st is SemanticType.ADDRESS:
        return pa.array([fake.address().replace("\n", ", ") for _ in range(count)])
    return pa.array([fake.text(max_nb_chars=80) for _ in range(count)])


def _mimesis_column(field: FieldSpec, count: int, rng_seed: int) -> pa.Array:
    from mimesis import Address as MimesisAddress
    from mimesis import Person as MimesisPerson
    from mimesis import Text as MimesisText
    from mimesis.locales import Locale as MimesisLocale

    locale_name = str(field.constraints.get("faker_locale") or "en").split("_")[0].upper()
    locale = getattr(MimesisLocale, locale_name, MimesisLocale.EN)
    st = field.semantic_type
    if st is SemanticType.NAME:
        person = MimesisPerson(locale, seed=rng_seed)
        return pa.array([person.full_name() for _ in range(count)])
    if st is SemanticType.EMAIL:
        person = MimesisPerson(locale, seed=rng_seed)
        return pa.array([person.email() for _ in range(count)])
    if st is SemanticType.ADDRESS:
        address = MimesisAddress(locale, seed=rng_seed)
        return pa.array([address.address() for _ in range(count)])
    text = MimesisText(locale, seed=rng_seed)
    return pa.array([text.sentence() for _ in range(count)])
