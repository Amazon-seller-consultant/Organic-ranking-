"""Errors surfaced to sellers. Every message must say exactly what is wrong
with *their* upload — never a stack trace, never a guessed mapping."""


class CatalogEngineError(Exception):
    """Base class for all engine errors."""


class FlatFileError(CatalogEngineError):
    """The uploaded file cannot be parsed as an Amazon Category Listing Report.

    `detail` is seller-facing; `hint` suggests what to do about it.
    """

    def __init__(self, detail: str, hint: str = ""):
        self.detail = detail
        self.hint = hint
        super().__init__(detail if not hint else f"{detail} — {hint}")


class SellerNotFoundError(CatalogEngineError):
    pass


class GenerationError(CatalogEngineError):
    """LLM generation failed for a SKU after retries."""

    def __init__(self, sku: str, detail: str):
        self.sku = sku
        super().__init__(f"generation failed for SKU {sku}: {detail}")
