from typing import Any

from pydantic import BaseModel, model_validator
from qdrant_client.http import models as qmodels


class MetadataFilter(BaseModel):
    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @model_validator(mode="after")
    def normalize(self) -> "MetadataFilter":
        names = [n.strip() for n in (self.filenames or []) if isinstance(n, str) and n.strip()]
        if not names:
            self.filenames = None
        elif len(names) == 1:
            self.filename, self.filenames = names[0], None
        else:
            self.filename, self.filenames, self.page = None, names, None

        for field in ("filename", "section", "document_id"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.strip() or None)
        return self


def _coerce_filter(filters: MetadataFilter | dict[str, Any] | None) -> MetadataFilter | None:
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        return filters
    return MetadataFilter.model_validate(filters)


def filters_to_dict(filters: MetadataFilter | dict[str, Any] | None) -> dict[str, Any] | None:
    coerced = _coerce_filter(filters)
    return None if coerced is None else coerced.model_dump(exclude_none=True) or None


def filters_to_qdrant(filters: MetadataFilter | dict[str, Any] | None) -> qmodels.Filter | None:
    flat = filters_to_dict(filters)
    if not flat:
        return None

    conditions: list[qmodels.FieldCondition] = []
    for field, value in flat.items():
        if field == "filenames" and isinstance(value, list):
            conditions.append(
                qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchAny(any=value))
            )
        elif isinstance(value, str | int):
            conditions.append(
                qmodels.FieldCondition(
                    key=f"metadata.{field}", match=qmodels.MatchValue(value=value)
                )
            )
    return qmodels.Filter(must=conditions) if conditions else None
