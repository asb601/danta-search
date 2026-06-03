"""Stage 8 — Parser Router.

A pure routing function (no external library needed). It reads a page's
``parser_hint`` (computed at preflight, stored on the page manifest) and selects
the concrete parser that will run in the worker.

| parser_hint          | parser       | when                                    |
|----------------------|--------------|-----------------------------------------|
| native               | PyMuPDF      | clean digital text PDF                  |
| complex_layout       | Docling      | multi-column / research / complex table |
| scanned              | Unstructured | scanned image pages needing OCR         |
| high_image_entropy   | VLM (gpt-4o-mini vision) | diagrams, charts, hand-drawn figures    |

``image_entropy`` is an additional override: even when the hint says ``native``
(or anything else), a page whose image entropy exceeds the configured threshold
is escalated to the VLM path, since high-entropy imagery is not reliably handled
by text/OCR parsers.
"""
from __future__ import annotations

from ..config import get_pdf_settings
from ..models.enums import ParserHint, ParserName

# Static hint → parser map. Adding a parser is a registry change, not code.
_HINT_TO_PARSER: dict[str, ParserName] = {
    ParserHint.NATIVE.value: ParserName.PYMUPDF,
    ParserHint.COMPLEX_LAYOUT.value: ParserName.DOCLING,
    ParserHint.SCANNED.value: ParserName.UNSTRUCTURED,
    ParserHint.HIGH_IMAGE_ENTROPY.value: ParserName.VLM,
}


def route_parser(parser_hint: str, image_entropy: float = 0.0) -> ParserName:
    """Select the concrete :class:`ParserName` for a page.

    Args:
        parser_hint: a :class:`ParserHint` value (str or enum). Unknown hints
            default to the safe native PyMuPDF path.
        image_entropy: page image entropy in ``[0, 1]``. Above the configured
            ``image_entropy_vlm_threshold`` the page is routed to the VLM
            regardless of ``parser_hint``.

    Returns:
        The :class:`ParserName` the worker should invoke.
    """
    threshold = get_pdf_settings().image_entropy_vlm_threshold
    if image_entropy > threshold:
        return ParserName.VLM

    hint = parser_hint.value if isinstance(parser_hint, ParserHint) else str(parser_hint)
    return _HINT_TO_PARSER.get(hint, ParserName.PYMUPDF)
