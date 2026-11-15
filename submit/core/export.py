"""
Alias module for core.exporter to support both import forms:
`from core.exporter import CorpusExporter` and `from core.export import CorpusExporter`
"""
from core.exporter import CorpusExporter

__all__ = ["CorpusExporter"]
