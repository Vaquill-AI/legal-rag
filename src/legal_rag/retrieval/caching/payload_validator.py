"""Payload validation: check a dict against an injected schema before it is cached or sent.

The schema is data, so each deployment declares its own required fields (and any isolation
keys it uses) instead of the library hard-coding them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PayloadField:
    """One field of a payload schema."""

    name: str
    types: tuple[type, ...]
    required: bool = True
    nullable: bool = False


@dataclass(frozen=True)
class PayloadSchema:
    """The fields a payload must (or may) carry."""

    fields: Sequence[PayloadField]
    allow_extra: bool = True


@dataclass
class PayloadValidationResult:
    """Outcome of ``validate_payload``."""

    missing: list[str] = field(default_factory=list)
    wrong_type: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not (self.missing or self.wrong_type or self.unexpected)


class PayloadValidationError(ValueError):
    """Raised by ``assert_valid_payload``."""

    def __init__(self, label: str, result: PayloadValidationResult):
        self.result = result
        super().__init__(
            f"{label} invalid: missing={result.missing} wrong_type={result.wrong_type} "
            f"unexpected={result.unexpected}"
        )


def validate_payload(payload: Mapping[str, Any], schema: PayloadSchema) -> PayloadValidationResult:
    """Check ``payload`` against ``schema``."""
    result = PayloadValidationResult()
    known = {f.name for f in schema.fields}
    for f in schema.fields:
        if f.name not in payload:
            if f.required:
                result.missing.append(f.name)
            continue
        value = payload[f.name]
        if value is None:
            if not f.nullable:
                result.wrong_type.append(f.name)
        elif not isinstance(value, f.types):
            result.wrong_type.append(f.name)
    if not schema.allow_extra:
        result.unexpected = sorted(set(payload) - known)
    return result


def assert_valid_payload(
    payload: Mapping[str, Any], schema: PayloadSchema, *, label: str = "payload"
) -> None:
    """Raise ``PayloadValidationError`` unless the payload fits the schema."""
    result = validate_payload(payload, schema)
    if not result.is_valid:
        raise PayloadValidationError(label, result)
