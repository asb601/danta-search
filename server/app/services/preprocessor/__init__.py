"""Preprocessor sub-package.

Contains the type-detection registry and the pluggable cleaning-rule registry
used by data_preprocessor.

Public API — type detection:
    detect_column_converter(col_name, sample) -> ColumnConverter | None

Public API — cleaning rules:
    get_cleaning_profile(
        extra_null_patterns=(),
        extra_garbage_re_patterns=(),
    ) -> CleaningProfile
"""
from app.services.preprocessor.type_detection import (
    ColumnConverter,
    DEFAULT_REGISTRY,
    TypeDetectionRegistry,
    detect_column_converter,
)
from app.services.preprocessor.cleaning_rules import (
    CleaningProfile,
    NullPatternRule,
    GarbageKeywordRule,
    SeparatorRowRule,
    AllEmptyRowRule,
    get_cleaning_profile,
)

__all__ = [
    # type detection
    "ColumnConverter",
    "DEFAULT_REGISTRY",
    "TypeDetectionRegistry",
    "detect_column_converter",
    # cleaning rules
    "CleaningProfile",
    "NullPatternRule",
    "GarbageKeywordRule",
    "SeparatorRowRule",
    "AllEmptyRowRule",
    "get_cleaning_profile",
]
