"""
PORF — Publish Original Research Fast.

Multi-expert roundtable research with mind map knowledge accumulation
and a dedicated writer for polished long-form articles.

Example:
    from porf import research

    report = research("The rise of cyberpunk aesthetics in modern fashion")
    print(report.markdown)
"""

from .core import research, PROFILES, STYLES
from .encoder import Encoder
from .fetcher import enrich_sources
from .mind_map import MindMap, MindMapNode, Snippet, normalize_url
from .types import Report, Section, Source
from .search import (
    SearchEngine,
    TavilySearch,
    BraveSearch,
    SerperSearch,
    DuckDuckGoSearch,
    SearxNGSearch,
    get_search_engine,
)

__all__ = [
    "research",
    "PROFILES",
    "STYLES",
    "Encoder",
    "enrich_sources",
    "MindMap",
    "MindMapNode",
    "Snippet",
    "normalize_url",
    "Report",
    "Section",
    "Source",
    "SearchEngine",
    "TavilySearch",
    "BraveSearch",
    "SerperSearch",
    "DuckDuckGoSearch",
    "SearxNGSearch",
    "get_search_engine",
]

__version__ = "0.1.0"
