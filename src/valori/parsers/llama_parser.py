"""LlamaParser removed.

This module previously contained an AI-powered parser backed by the
`llama-parse` package. That implementation has been removed in favor of
Docling-based parsing. Importing this module will raise ImportError to
prevent accidental usage.
"""

raise ImportError(
    "LlamaParser has been removed from valori. Use the Docling-based parsers "
    "(see valori.parsers.docling_parser) instead."
)
