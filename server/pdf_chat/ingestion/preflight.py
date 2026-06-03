"""Stage 3 — Preflight / Repair (the "bouncer").

Runs cheap, fast checks before any expensive worker time is spent:
  * ``pikepdf``      — repair broken xref/linearization, detect encryption
  * ``pdfplumber``   — page count, scanned/digital classification, embedded fonts
  * ``python-magic`` — verify the file is genuinely a PDF by its magic bytes

The decision logic (``evaluate_preflight``) and per-page classification
(``classify_page``) are PURE and fully unit-testable with zero infra. The infra
libraries are imported behind guards (Hard rule #6): if any are missing,
``run_preflight`` degrades gracefully and still produces a ``PreflightResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import get_pdf_settings
from ..models.enums import ParserHint

# ---- Guarded infra imports (feature flags) --------------------------------
try:  # PDF repair + encryption detection
    import pikepdf  # type: ignore

    _HAS_PIKEPDF = True
except ImportError:  # pragma: no cover - exercised only without infra
    pikepdf = None  # type: ignore
    _HAS_PIKEPDF = False

try:  # page count / scanned classification / fonts
    import pdfplumber  # type: ignore

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore
    _HAS_PDFPLUMBER = False

try:  # magic-byte MIME sniffing
    import magic  # type: ignore  # python-magic

    _HAS_MAGIC = True
except ImportError:  # pragma: no cover
    magic = None  # type: ignore
    _HAS_MAGIC = False


PDF_MIME = "application/pdf"
# Magic bytes for a PDF file: "%PDF" (used when python-magic is unavailable).
_PDF_MAGIC = b"%PDF"


@dataclass
class PreflightResult:
    """Output stored as ``preflight_json`` on the upload manifest (Stage 3)."""

    is_encrypted: bool = False
    is_scanned: bool = False
    page_count: int = 0
    mime_type: str = ""
    has_embedded_fonts: bool = True
    repair_applied: bool = False
    scanned_pages: list[int] = field(default_factory=list)
    complex_layout_pages: list[int] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "is_encrypted": self.is_encrypted,
            "is_scanned": self.is_scanned,
            "page_count": self.page_count,
            "mime_type": self.mime_type,
            "has_embedded_fonts": self.has_embedded_fonts,
            "repair_applied": self.repair_applied,
            "scanned_pages": list(self.scanned_pages),
            "complex_layout_pages": list(self.complex_layout_pages),
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
        }


# ---- PURE decision logic ---------------------------------------------------
def evaluate_preflight(
    *,
    mime_type: str,
    is_encrypted: bool,
    page_count: int,
) -> tuple[bool, str | None]:
    """Decide whether a document is rejected at the gate.

    Reject rules (any one trips rejection):
      * the file is encrypted / password protected
      * the MIME type is not ``application/pdf`` (named ``.pdf`` but isn't one)
      * the page count is zero (empty / unreadable document)

    Returns ``(rejected, reject_reason)``. ``reject_reason`` is ``None`` when the
    document passes.
    """
    if is_encrypted:
        return True, "encrypted"
    if mime_type != PDF_MIME:
        return True, f"not_a_pdf:{mime_type or 'unknown'}"
    if page_count == 0:
        return True, "zero_pages"
    return False, None


def classify_page(
    text_char_count: int,
    *,
    scanned_char_threshold: int | None = None,
    is_complex_layout: bool = False,
    image_entropy: float = 0.0,
    entropy_threshold: float | None = None,
) -> ParserHint:
    """Classify a single page into a routing ``ParserHint`` (Stages 3/6).

    Order of precedence:
      1. ``image_entropy`` above the configured threshold → HIGH_IMAGE_ENTROPY
         (diagrams / charts / hand-drawn figures → VLM fallback downstream).
      2. text char count below the scanned threshold → SCANNED (needs OCR).
      3. explicit complex-layout flag → COMPLEX_LAYOUT.
      4. otherwise → NATIVE (clean digital text).

    Thresholds come from ``PdfSettings`` (config-driven, no magic numbers).
    """
    settings = get_pdf_settings()
    if scanned_char_threshold is None:
        scanned_char_threshold = settings.scanned_text_char_threshold
    if entropy_threshold is None:
        entropy_threshold = settings.image_entropy_vlm_threshold

    if image_entropy > entropy_threshold:
        return ParserHint.HIGH_IMAGE_ENTROPY
    if text_char_count < scanned_char_threshold:
        return ParserHint.SCANNED
    if is_complex_layout:
        return ParserHint.COMPLEX_LAYOUT
    return ParserHint.NATIVE


# ---- Infra-backed orchestration (graceful degradation) ---------------------
def _sniff_mime(file_bytes: bytes) -> str:
    """Best-effort MIME detection. Falls back to magic-byte sniffing."""
    if _HAS_MAGIC:
        try:
            return magic.from_buffer(file_bytes[:2048], mime=True)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - defensive
            pass
    # Pure fallback: inspect the leading bytes ourselves.
    return PDF_MIME if file_bytes[:4] == _PDF_MAGIC else "application/octet-stream"


def _detect_encryption_and_repair(file_bytes: bytes) -> tuple[bool, bool]:
    """Return ``(is_encrypted, repair_applied)`` using pikepdf when available."""
    if not _HAS_PIKEPDF:
        return False, False
    import io

    try:
        pikepdf.open(io.BytesIO(file_bytes))  # type: ignore[union-attr]
        return False, False
    except pikepdf.PasswordError:  # type: ignore[union-attr]
        return True, False
    except Exception:
        # Broken xref/linearization — attempt an in-memory repair pass.
        try:
            buf = io.BytesIO()
            with pikepdf.open(io.BytesIO(file_bytes), allow_overwriting_input=True) as pdf:  # type: ignore[union-attr]
                pdf.save(buf)
            return False, True
        except pikepdf.PasswordError:  # type: ignore[union-attr]
            return True, False
        except Exception:  # pragma: no cover - repair failed
            return False, False


def _analyze_pages(file_bytes: bytes) -> tuple[int, list[int], bool]:
    """Return ``(page_count, scanned_pages, has_embedded_fonts)`` via pdfplumber."""
    if not _HAS_PDFPLUMBER:
        return 0, [], True
    import io

    settings = get_pdf_settings()
    scanned_pages: list[int] = []
    has_fonts = True
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:  # type: ignore[union-attr]
            pages = pdf.pages
            page_count = len(pages)
            for idx, page in enumerate(pages):
                text = page.extract_text() or ""
                hint = classify_page(
                    len(text),
                    scanned_char_threshold=settings.scanned_text_char_threshold,
                )
                if hint == ParserHint.SCANNED:
                    scanned_pages.append(idx)
                # Missing fonts on any page flags an OCR-quality warning.
                if not getattr(page, "chars", None):
                    has_fonts = False
            return page_count, scanned_pages, has_fonts
    except Exception:  # pragma: no cover - defensive
        return 0, [], True


def run_preflight(file_bytes: bytes) -> PreflightResult:
    """Run the full preflight gate over raw PDF bytes.

    Uses pikepdf / pdfplumber / python-magic via guarded imports. Any missing
    library degrades gracefully (the corresponding signal is left at its safe
    default) so the function never raises on a clean machine without infra. The
    accept/reject decision is delegated to the pure ``evaluate_preflight``.
    """
    mime_type = _sniff_mime(file_bytes)
    is_encrypted, repair_applied = _detect_encryption_and_repair(file_bytes)
    page_count, scanned_pages, has_fonts = _analyze_pages(file_bytes)

    rejected, reason = evaluate_preflight(
        mime_type=mime_type,
        is_encrypted=is_encrypted,
        page_count=page_count,
    )

    return PreflightResult(
        is_encrypted=is_encrypted,
        is_scanned=bool(scanned_pages),
        page_count=page_count,
        mime_type=mime_type,
        has_embedded_fonts=has_fonts,
        repair_applied=repair_applied,
        scanned_pages=scanned_pages,
        complex_layout_pages=[],
        rejected=rejected,
        reject_reason=reason,
    )
