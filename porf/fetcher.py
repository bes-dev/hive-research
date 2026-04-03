"""Fetch full page content for key sources via trafilatura."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .types import Source


def _fetch_page(url: str, timeout: int = 15) -> str | None:
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        html = trafilatura.fetch_url(url)
        if not html:
            return None
        return trafilatura.extract(html, include_links=False, include_comments=False)
    except Exception:
        return None


def enrich_sources(sources: list[Source], max_sources: int = 8,
                   max_chars: int = 3000) -> int:
    """Fetch full page content for top sources, mutating .content in-place.

    Returns count of successfully enriched sources.
    """
    targets = sources[:max_sources]
    if not targets:
        return 0
    enriched = 0
    with ThreadPoolExecutor(max_workers=min(len(targets), 5)) as pool:
        futures = {pool.submit(_fetch_page, src.url): src for src in targets}
        for future in as_completed(futures):
            src = futures[future]
            try:
                text = future.result()
                if text and len(text) > len(src.content):
                    src.content = text[:max_chars]
                    enriched += 1
            except Exception:
                pass
    return enriched
