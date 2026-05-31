"""ERP knowledge layer.

This package adds the BUSINESS and OPERATIONAL context layers that a pure
retrieval pipeline lacks: which ERP system a file belongs to, which functional
module, which side of the ledger (customer vs vendor), and where it sits on a
business process chain.

Design rules (non-negotiable — see ARCHITECTURE_DEEP_DIVE / response.txt):
  • NO hardcoded ERP dictionaries. Classification is DATA-DRIVEN: it uses the
    LLM's general ERP knowledge plus evidence already extracted at ingestion
    (column names, sample values, semantic roles, AI description). This means it
    generalises to ANY client's archived business files — SAP, Oracle EBS,
    NetSuite, Dynamics, Workday, or a bespoke schema — not a fixed vendor list.
  • Every classification carries a confidence score and human-readable evidence.
  • Low confidence degrades to "unknown"/"neutral" and NEVER blocks the pipeline
    or forces a wrong scoping decision. Unknown == behave like today.
  • Results are persisted with provenance and are human-overridable.
"""

from app.services.erp.classifier import (  # noqa: F401
    ErpClassification,
    ErpClassifier,
    Polarity,
    classify_file,
)
