"""The Danta Semantic Contract (DSC) — our ERP-aware semantic layer.

A governed projection over existing ingestion artifacts that the planner and
the dry-plan gate reason against, instead of raw files. Joins are DECLARED
(approved-only); columns are EXPOSED explicitly; business meaning (source
system, module, polarity, process role, value semantics) travels with each
model. See response.txt / ARCHITECTURE_DEEP_DIVE.txt for the full rationale.
"""

from app.services.contract.builder import (  # noqa: F401
    build_contract,
    compile_and_store_contract,
    load_contract,
)
from app.services.contract.dry_plan import DryPlanVerdict, dry_plan_sql  # noqa: F401
