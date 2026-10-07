"""One common document format for every source (CSV now; PDF, API, web later)."""
import uuid
from dataclasses import dataclass, field

_NAMESPACE = uuid.UUID("6f1c2f0e-8d3a-4c57-9b1e-2a5d7e0c9a11")


@dataclass
class Document:
    doc_id: str                      # stable id, e.g. "LAP001" or "LAP001#pdf#3"
    text: str                        # the text that gets embedded
    source_type: str                 # "csv" | "pdf" | "api" | "web"
    source_name: str
    product_id: str | None = None    # links every chunk back to a laptop
    fetched_at: str | None = None
    meta: dict = field(default_factory=dict)  # structured fields used as filters

    @property
    def point_id(self) -> str:
        """Qdrant needs int/UUID ids; derive a deterministic UUID so re-indexing overwrites."""
        return str(uuid.uuid5(_NAMESPACE, self.doc_id))

    def to_payload(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "product_id": self.product_id,
            "fetched_at": self.fetched_at,
            **self.meta,
        }
