"""PORF core — multi-expert roundtable research + writer pipeline."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable

import litellm
litellm.suppress_debug_info = True

from . import prompts as _prompts_module
from .encoder import Encoder, cosine_similarity
from .fetcher import enrich_sources
from .mind_map import MindMap, MindMapNode, Snippet, normalize_url
from .search import SearchEngine, get_search_engine
from .types import Report, Section, Source

# ── Utilities ────────────────────────────────────────────────────────────

LLM = Callable[..., str]


def _resolve_models(model: str | dict) -> dict[str, str]:
    """Normalize model spec into {default, fast, review} dict."""
    if isinstance(model, str):
        return {"default": model, "fast": model, "review": model}
    default = model.get("default", "anthropic/claude-sonnet-4-20250514")
    return {
        "default": default,
        "fast": model.get("fast", default),
        "review": model.get("review", default),
    }


def _make_llm(model: str, api_base: str | None, usage: dict,
              lock: threading.Lock, trace: list | None) -> LLM:
    return partial(_llm_call, model, api_base=api_base,
                   _usage=usage, _usage_lock=lock,
                   _trace=trace, _trace_lock=lock)


def _llm_call(model: str, prompt: str, api_base: str | None = None,
              _usage: dict | None = None, _usage_lock: threading.Lock | None = None,
              _trace: list | None = None, _trace_lock: threading.Lock | None = None,
              max_retries: int = 5, max_tokens: int | None = None,
              response_format: dict | None = None,
              timeout: int = 300) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9, "timeout": timeout,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if response_format:
        kwargs["response_format"] = response_format
    if api_base:
        kwargs["api_base"] = api_base

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(**kwargs)
            content = resp.choices[0].message.content

            pt = ct = 0
            if u := getattr(resp, "usage", None):
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0

            if _usage:
                if _usage_lock:
                    with _usage_lock:
                        _usage["prompt_tokens"] += pt
                        _usage["completion_tokens"] += ct
                else:
                    _usage["prompt_tokens"] += pt
                    _usage["completion_tokens"] += ct

            if _trace is not None:
                entry = {
                    "timestamp": time.time(), "model": model,
                    "prompt": prompt, "response": content,
                    "json_mode": response_format is not None,
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct},
                }
                if _trace_lock:
                    with _trace_lock:
                        _trace.append(entry)
                else:
                    _trace.append(entry)

            return content
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            if (s := getattr(e, "status_code", None)) and 400 <= s < 500 and s != 429:
                raise
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 30))
    raise last_error


def _parse_json(text: str, expect: str = "array"):
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', text).strip()
    result = None
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        brackets = [('[', ']'), ('{', '}')] if expect == "array" else [('{', '}'), ('[', ']')]
        for ob, cb in brackets:
            start = cleaned.find(ob)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == ob:
                    depth += 1
                elif cleaned[i] == cb:
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
            if result is not None:
                break
    if result is None:
        return {} if expect == "object" else []
    if expect == "object" and isinstance(result, list):
        return {}
    if expect == "array" and isinstance(result, dict):
        return []
    return result


def _get_prompt(name: str, ov: dict | None = None) -> str:
    return (ov or {}).get(name) or getattr(_prompts_module, name)


# ── Styles & Profiles ───────────────────────────────────────────────────

STYLES = {
    "academic": {
        "narrative": "Structure as an academic review: build from definitions to evidence to synthesis.",
        "techniques": (
            "Open sections with a precise thesis statement grounded in evidence. "
            "Build arguments through logical chains: claim → evidence → interpretation. "
            "Acknowledge counterarguments explicitly before addressing them. "
            "Use field-specific terminology naturally. "
            "End sections by connecting findings to the broader theoretical framework."
        ),
        "tone": "Measured, precise, authoritative but not pompous.",
    },
    "popular": {
        "narrative": (
            "Structure as a popular-science longread: open with a person, scene or "
            "dramatic moment, then peel back layers — each section goes deeper but "
            "stays accessible. Build narrative tension: pose a question early, delay "
            "the answer. Use multiple viewpoints to enrich the story."
        ),
        "techniques": (
            "ZERO JARGON. The reader is a curious non-expert — a bank clerk, a "
            "barista, a retired teacher. If a term wouldn't be obvious to them, "
            "don't use it. Never write 'simulacra of the third order' — write "
            "'copies that no longer have an original, like a photo of a photo of "
            "a photo until nobody remembers what was real'. "
            "Feynman rule: explain every idea as if the reader has zero background "
            "but full intelligence. Use analogies from everyday life — kitchens, "
            "traffic, sports, weather — not from other theories. "
            "People over abstractions: anchor each section in a specific person, "
            "place, or moment. 'In 2019, a team in Geneva...' beats 'Research shows...'. "
            "Drama: create tension by showing what's at stake before explaining how "
            "it works. Surprise the reader — counterintuitive facts, unexpected turns. "
            "Short paragraphs. One-sentence paragraphs for punch. "
            "Mix humor or irony with substance — but never be cute at the expense "
            "of clarity. "
            "Skip theorist names entirely unless the person IS the story. Never "
            "write 'as Baudrillard argued' — just explain the idea in plain words. "
            "One complex idea per section max. If the research draft has five "
            "theories, pick the one that best serves the story, drop the rest."
        ),
        "tone": (
            "Warm, curious, zero academic register. A science journalist at a "
            "dinner party — smart, funny, makes you feel clever for following along. "
            "The reader should think 'huh, I never knew that' not 'I need a PhD "
            "to parse this sentence'."
        ),
    },
    "journalistic": {
        "narrative": "Investigative long-form: lead with the most striking finding, build context, present multiple viewpoints, end with implications.",
        "techniques": (
            "Lead with the most striking fact or scene — who, where, when. "
            "Build context immediately after the hook. "
            "Alternate between evidence and human stories. "
            "Use short, punchy paragraphs. "
            "End with implications or what comes next. "
            "Use theory sparingly — always lead with the concrete fact, "
            "then add the theoretical frame only if it adds explanatory power."
        ),
        "tone": "Direct, concrete, matter-of-fact with controlled urgency.",
    },
    "analytical": {
        "narrative": "Systematic analysis: define the problem space, identify key factors, weigh evidence, synthesize conclusions.",
        "techniques": (
            "Open sections by defining the problem or question. "
            "Present multiple factors systematically. "
            "Use data and specific numbers wherever possible. "
            "Compare competing explanations fairly. "
            "End with a synthesis that weighs the evidence."
        ),
        "tone": "Clear-headed, systematic, fair — like a skilled analyst briefing a decision-maker.",
    },
    "essay": {
        "narrative": "Thoughtful essay: open with a personal observation or question, explore through evidence and reflection, arrive at a synthesis.",
        "techniques": (
            "Open with a personal observation, memory, or question. "
            "Let the argument develop through exploration, not assertion. "
            "Allow tangents that illuminate the main theme. "
            "Use metaphors and analogies drawn from everyday life. "
            "End with a moment of synthesis that feels earned, not imposed. "
            "Introduce theories through personal engagement — "
            "'I keep returning to this phrase...' rather than 'X argued that...'."
        ),
        "tone": "Reflective, intellectually honest, personal — thinking on the page.",
    },
}

PROFILES = {
    "quick": {
        "n_perspectives": 3, "max_rounds": 10,
        "max_results": 3, "max_queries": 2,
        "lock_sources": 5, "target_words": 2000,
        "review_cycles": 0,
    },
    "balanced": {
        "n_perspectives": 4, "max_rounds": 15,
        "max_results": 5, "max_queries": 2,
        "lock_sources": 8, "target_words": 4000,
        "review_cycles": 1,
    },
    "deep": {
        "n_perspectives": 5, "max_rounds": 20,
        "max_results": 8, "max_queries": 3,
        "lock_sources": 12, "target_words": 6000,
        "review_cycles": 2,
    },
}


def _domain_diversify(results: list[Source], max_per_domain: int = 2) -> list[Source]:
    """Limit results per domain to improve source diversity."""
    from urllib.parse import urlparse
    counts: dict[str, int] = {}
    selected: list[Source] = []
    for r in results:
        domain = (urlparse(r.url).hostname or '').removeprefix('www.')
        if counts.get(domain, 0) < max_per_domain:
            selected.append(r)
            counts[domain] = counts.get(domain, 0) + 1
    return selected or results


_BLOCKED_DOMAINS = {
    'pinterest.com', 'pinterest.ru', 'instagram.com', 'facebook.com',
    'twitter.com', 'x.com', 'tiktok.com', 'vk.com', 'ok.ru',
    'sportmaster.ru', 'wildberries.ru', 'ozon.ru', 'amazon.com',
    'aliexpress.com', 'ebay.com', 'etsy.com',
    'youtube.com', 'youtu.be', 'twitch.tv',
}


def _filter_blocked_domains(results: list[Source]) -> list[Source]:
    """Remove results from social media, e-commerce, and other non-research domains."""
    from urllib.parse import urlparse
    filtered = []
    for r in results:
        host = (urlparse(r.url).hostname or '').removeprefix('www.').lower()
        if host not in _BLOCKED_DOMAINS:
            filtered.append(r)
    return filtered or results


def _filter_sources(results: list[Source], topic: str,
                    llm: LLM, ov: dict | None) -> list[Source]:
    """Filter search results by LLM-judged relevance to topic."""
    if len(results) <= 2:
        return results
    sources_text = "\n".join(
        f"{i+1}. [{r.title}] {r.url} — {r.content[:300]}"
        for i, r in enumerate(results))
    try:
        raw = llm(_get_prompt("FILTER_SOURCES", ov).format(
            topic=topic, sources=sources_text,
        ), max_tokens=300, response_format={"type": "json_object"})
        ratings = _parse_json(raw, expect="object").get("ratings", [])
        keep = {r.get("id", 0) for r in ratings if r.get("score", 0) >= 2}
        filtered = [r for i, r in enumerate(results) if (i + 1) in keep]
        return filtered or results
    except Exception:
        return results


# ── Information routing ──────────────────────────────────────────────────

def _route_snippet(mm: MindMap, snippet: Snippet,
                   encoder: Encoder, llm: LLM, ov: dict | None) -> MindMapNode:
    """Find the best mind map node for a snippet (embedding rank → LLM fallback)."""
    paths = mm.structure_paths()
    if not paths:
        return mm.root

    if encoder.available():
        try:
            query_text = f"{snippet.question} {snippet.query}"
            path_strings = [p for _, p in paths]
            embs = encoder.encode([query_text] + path_strings)
            q_emb, path_embs = embs[0], embs[1:]

            scores = sorted(enumerate(path_embs),
                            key=lambda x: cosine_similarity(q_emb, x[1]),
                            reverse=True)[:8]

            candidates = "\n".join(
                f"{j + 1}. {paths[i][1]}" for j, (i, _) in enumerate(scores))
            choice = llm(_get_prompt("INSERT_CHOOSE", ov).format(
                question=snippet.question,
                snippet=snippet.source.content[:500],
                candidates=candidates,
            )).strip()

            if choice.lower() != "none":
                idx = int(re.search(r'\d+', choice).group()) - 1
                if 0 <= idx < len(scores):
                    return paths[scores[idx][0]][0]
        except Exception:
            pass

    return _llm_navigate(mm, snippet, llm, ov)


def _llm_navigate(mm: MindMap, snippet: Snippet,
                  llm: LLM, ov: dict | None,
                  allow_create: bool = True) -> MindMapNode:
    """Navigate mind map tree to place a snippet."""
    node = mm.root
    for _ in range(10):
        if not node.children:
            return node
        children_text = "\n".join(
            f"- {c.name}" + (f": {c.brief}" if c.brief else "")
            for c in node.children)
        result = llm(_get_prompt("INSERT_NAVIGATE", ov).format(
            question=f"{snippet.question} {snippet.query}",
            current_node=node.name, children=children_text,
        )).strip()

        low = result.lower()
        if low == "insert":
            return node
        if low.startswith("step:"):
            target = result[5:].strip().strip('"').strip("'")
            found = None
            for c in node.children:
                if c.name.lower() == target.lower():
                    found = c
                    break
            if not found:
                for c in node.children:
                    if target.lower() in c.name.lower() or c.name.lower() in target.lower():
                        found = c
                        break
            node = found or node
            if not found:
                return node
        elif low.startswith("create:") and allow_create:
            new_name = result[7:].strip().strip('"').strip("'")
            return mm.add_child(node, new_name)
        else:
            return node
    return node


def _insert_turn_snippets(mm: MindMap, turn: dict,
                          encoder: Encoder, llm: LLM,
                          ov: dict | None) -> int:
    """Insert cited sources into mind map. Returns count inserted."""
    results = turn.get("results", [])
    cited = turn.get("cited_indices", set())
    question = turn.get("question", "")
    queries = turn.get("queries", [])
    query_str = queries[0] if queries else question
    inserted = 0

    target_node = turn.get("target_node")
    for idx in cited:
        if 0 < idx <= len(results):
            src = results[idx - 1]
            snippet = Snippet(source=src, question=question, query=query_str)
            uid = mm.add_snippet(snippet)
            if uid is not None:
                if target_node:
                    mm.attach(target_node, uid)
                else:
                    target = _route_snippet(mm, snippet, encoder, llm, ov)
                    mm.attach(target, uid)
                inserted += 1

    mm.add_to_reservoir(results)
    return inserted


# ── Roundtable: research = drafting ──────────────────────────────────────

def _apply_restructure(mm: MindMap, changes: list, index_map: dict[int, 'MindMapNode'],
                       log: Callable):
    """Apply restructure actions (reword/merge/split/add/drop) to MindMap."""

    def _resolve(ref) -> MindMapNode | None:
        """Resolve section reference: int number or string name."""
        if isinstance(ref, (int, float)):
            return index_map.get(int(ref))
        if isinstance(ref, str):
            m = re.match(r'^(\d+)', ref.strip())
            if m:
                return index_map.get(int(m.group(1)))
            return next((n for n in mm.root.all_nodes()
                         if n is not mm.root and n.name == ref), None)
        return None

    for change in changes:
        if not isinstance(change, dict):
            log(f"      skip non-dict restructure: {str(change)[:80]}")
            continue
        action = change.get("action", "")
        if action == "reword":
            node = _resolve(change.get("section"))
            if node:
                node.claim = change.get("new_claim", node.claim)
                log(f"      reword [{node.name}]: {node.claim[:60]}")

        elif action == "rename":
            node = _resolve(change.get("section"))
            new_name = change.get("name", "")
            if node and new_name:
                old_name = node.name
                node.name = new_name
                node.claim = change.get("claim", node.claim)
                log(f"      rename [{old_name}] → [{new_name}]")

        elif action == "merge":
            refs = change.get("sections", [])
            nodes = [n for r in refs if (n := _resolve(r))]
            if len(nodes) >= 2:
                merged = nodes[0]
                merged.name = change.get("into", merged.name)
                merged.claim = change.get("claim", merged.claim)
                for other in nodes[1:]:
                    merged.snippet_uuids |= other.snippet_uuids
                    merged.draft += "\n\n" + other.draft if other.draft else ""
                    if other.parent:
                        other.parent.children = [
                            c for c in other.parent.children if c is not other]
                log(f"      merge → [{merged.name}]")

        elif action == "split":
            node = _resolve(change.get("section"))
            into = change.get("into", [])
            if node and into and not node.children:
                for sub in into:
                    sub_name = sub.get("name", "")
                    if sub_name:
                        child = mm.add_child(node, sub_name)
                        child.claim = sub.get("claim", "")
                uuids = list(node.snippet_uuids)
                node.snippet_uuids.clear()
                for i, uid in enumerate(uuids):
                    if node.children:
                        node.children[i % len(node.children)].snippet_uuids.add(uid)
                log(f"      split [{node.name}] → {[s.get('name', '') for s in into]}")

        elif action == "add":
            name = change.get("name", "")
            if name:
                new_node = mm.add_child(mm.root, name)
                new_node.claim = change.get("claim", "")
                log(f"      add [{name}]")

        elif action == "drop":
            node = _resolve(change.get("section"))
            if node and node.parent:
                for uid in node.snippet_uuids:
                    if uid in mm.snippets:
                        mm.reservoir.append(mm.snippets[uid].source)
                node.parent.children = [
                    c for c in node.parent.children if c is not node]
                log(f"      drop [{node.name}]")


def _assign_section(mm: MindMap, locked_nodes: set[MindMapNode]) -> MindMapNode | None:
    """Pick the least-explored unlocked leaf section."""
    candidates = [leaf for leaf in mm.root.leaves() if leaf not in locked_nodes]
    if not candidates:
        return None
    return min(candidates, key=lambda n: len(n.snippet_uuids))


def _unique_domains(mm: MindMap, node: MindMapNode) -> int:
    """Count unique domains among a node's snippets."""
    from urllib.parse import urlparse
    domains: set[str] = set()
    for uid in node.snippet_uuids:
        if uid in mm.snippets:
            host = (urlparse(mm.snippets[uid].source.url).hostname or '').removeprefix('www.')
            if host:
                domains.add(host)
    return len(domains)


def _check_convergence(mm: MindMap, lock_threshold: int,
                       locked_nodes: set[MindMapNode]) -> set[MindMapNode]:
    """Auto-lock leaves that have enough sources from diverse domains AND a draft."""
    min_domains = 3
    for leaf in mm.root.leaves():
        if leaf in locked_nodes:
            continue
        # Subsections (depth > 1) converge faster — they cover narrower topics
        depth = 0
        node = leaf
        while node.parent:
            depth += 1
            node = node.parent
        threshold = max(3, lock_threshold // depth) if depth > 1 else lock_threshold
        if not leaf.draft or len(leaf.snippet_uuids) < threshold:
            continue
        if _unique_domains(mm, leaf) >= min(min_domains, threshold):
            locked_nodes.add(leaf)
    return locked_nodes


def _section_outline(mm: MindMap, locked_nodes: set[MindMapNode]) -> tuple[dict[int, MindMapNode], str]:
    """Compact outline for RESEARCH_TURN prompt."""
    index_map: dict[int, MindMapNode] = {}
    lines: list[str] = []
    counter = 0

    def _walk(node: MindMapNode, indent: int):
        nonlocal counter
        for child in node.children:
            counter += 1
            index_map[counter] = child
            n = len(child.snippet_uuids)
            locked = "[LOCKED] " if child in locked_nodes else ""
            claim = f" — {child.claim}" if child.claim else ""
            lines.append(f"{'  ' * indent}{counter}. {locked}{child.name} ({n} src){claim}")
            _walk(child, indent + 1)

    _walk(mm.root, 0)
    return index_map, "\n".join(lines)


def _research_turn(topic: str, agent: dict, section: MindMapNode,
                   mm: MindMap, prof: dict, search_lang: str, output_lang: str,
                   engine: SearchEngine, llm: LLM, llm_fast: LLM,
                   ov: dict | None, log: Callable) -> dict:
    """One research turn: generate queries → search → draft section."""
    role = agent["role"]
    desc = agent.get("description", "")

    log(f"    [{role}] → \"{section.name}\"")

    try:
        return _research_turn_inner(
            topic, role, desc, section, mm, prof, search_lang,
            output_lang, engine, llm, llm_fast, ov, log)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log(f"      error (skipping turn): {e}")
        return {"role": role, "section": section.name,
                "queries": [], "results": [], "cited_indices": set(),
                "target_node": section, "question": section.name,
                "restructure": [], "index_map": {}}


def _research_turn_inner(topic: str, role: str, desc: str, section: MindMapNode,
                         mm: MindMap, prof: dict, search_lang: str, output_lang: str,
                         engine: SearchEngine, llm: LLM, llm_fast: LLM,
                         ov: dict | None, log: Callable) -> dict:
    # 1. Build outline
    index_map, outline = _section_outline(mm, set())

    # 2. Generate search queries
    oq = ", ".join(section.open_questions) if section.open_questions else "(none)"
    raw = llm_fast(_get_prompt("RESEARCH_TURN", ov).format(
        role=role, description=desc, topic=topic,
        outline=outline,
        section_name=section.name,
        section_claim=section.claim or "(none)",
        section_draft=section.draft or "(empty)",
        n_sources=len(section.snippet_uuids),
        open_questions=oq,
        search_lang_instruction=search_lang,
        max_queries=prof["max_queries"],
    ), response_format={"type": "json_object"})
    queries = _parse_json(raw, expect="object").get("queries", [])
    if not queries:
        queries = [f"{section.name} {section.claim}"]

    # 3. Search + filter
    log(f"      queries: {queries}")
    all_results: list[Source] = []
    for q in queries[:prof["max_queries"]]:
        try:
            all_results.extend(engine.search(q, max_results=prof["max_results"]))
        except Exception:
            pass
    raw_count = len(all_results)
    all_results = _filter_blocked_domains(all_results)
    all_results = _domain_diversify(all_results)
    all_results = _filter_sources(all_results, topic, llm_fast, ov)
    log(f"      search: {raw_count} raw → {len(all_results)} after filter")

    cited: set[int] = set()

    if all_results:
        # 4. Draft section with new evidence
        sources_text = "\n".join(
            f"[{i}] {s.title}\n{s.content}" for i, s in enumerate(all_results, 1))
        raw = llm(_get_prompt("DRAFT_SECTION", ov).format(
            role=role, description=desc,
            section_name=section.name,
            section_claim=section.claim or "(none)",
            section_draft=section.draft or "(empty — write the first draft)",
            sources=sources_text,
            output_lang_instruction=output_lang,
        ), response_format={"type": "json_object"})
        result = _parse_json(raw, expect="object")

        # 5. Update node
        new_claim = result.get("claim", "")
        new_draft = result.get("draft", "")
        new_oq = result.get("open_questions", [])

        if new_claim:
            section.claim = new_claim
        if new_draft:
            section.draft = new_draft
        if isinstance(new_oq, list):
            section.open_questions = new_oq
        section.contributors.add(role)

        cited = {int(m) for m in re.findall(r'\[(\d+)\]', new_draft)}
        log(f"      cited: {len(cited)}/{len(all_results)} sources")
        if new_claim:
            log(f"      claim: {new_claim[:80]}")

        # 6. Collect restructure proposals (applied after parallel turns)
        restructure = result.get("restructure") or []
    else:
        restructure = []
        log(f"      no results")

    return {
        "role": role, "section": section.name,
        "queries": queries, "results": all_results, "cited_indices": cited,
        "target_node": section,
        "question": section.claim or section.name,
        "restructure": restructure if isinstance(restructure, list) else [],
        "index_map": index_map,
    }


def _roundtable(topic: str, perspectives: list[dict], mm: MindMap,
                prof: dict, search_lang: str, output_lang: str,
                engine: SearchEngine, encoder: Encoder,
                llm: LLM, llm_fast: LLM,
                ov: dict | None, log: Callable):
    """Roundtable: each expert researches and drafts assigned sections."""
    max_rounds = prof["max_rounds"]
    lock_threshold = prof["lock_sources"]
    locked_nodes: set[MindMapNode] = set()

    log(f"Roundtable: up to {max_rounds} rounds, {len(perspectives)} experts...")
    log(f"  Experts: {', '.join(p['role'] for p in perspectives)}")

    for rnd in range(max_rounds):
        log(f"  Round {rnd + 1}/{max_rounds}...")

        # Pre-assign different sections to each expert
        used: set[MindMapNode] = set()
        assignments: list[tuple[dict, MindMapNode]] = []
        for agent in perspectives:
            section = _assign_section(mm, locked_nodes | used)
            if section is None:
                continue
            assignments.append((agent, section))
            used.add(section)

        if not assignments:
            log("  All sections converged!")
            break

        # Run expert turns in parallel
        turns: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            futures = {
                pool.submit(
                    _research_turn, topic, agent, section, mm, prof,
                    search_lang, output_lang, engine, llm, llm_fast, ov, log,
                ): section.name
                for agent, section in assignments
            }
            for future in as_completed(futures):
                turns.append(future.result())

        # Apply results sequentially: insert snippets + restructures
        for turn in turns:
            if turn.get("results"):
                inserted = _insert_turn_snippets(mm, turn, encoder, llm_fast, ov)
                log(f"      +{inserted} snippets [{turn['section']}]")
            restructure = turn.get("restructure", [])
            if restructure:
                log(f"      restructure: {len(restructure)} changes [{turn['section']}]")
                _apply_restructure(mm, restructure, turn.get("index_map", {}), log)

        _check_convergence(mm, lock_threshold, locked_nodes)
        n_leaves = len(mm.root.leaves())
        log(f"  {len(locked_nodes)}/{n_leaves} locked")

    mm.trim()

    n_rounds_done = rnd + 1
    log(f"  Roundtable done: {n_rounds_done} rounds, "
        f"{len(mm.root.leaves())} sections, {len(mm.snippets)} snippets, "
        f"{len(locked_nodes)} locked")
    for leaf in mm.root.leaves():
        claim_preview = f" — {leaf.claim[:60]}..." if leaf.claim else ""
        draft_len = f", {len(leaf.draft)} chars draft" if leaf.draft else ""
        log(f"    [{leaf.name}] {len(leaf.snippet_uuids)} src{draft_len}{claim_preview}")


# ── Debug render ─────────────────────────────────────────────────────────

def _render_debug(topic: str, mm: MindMap) -> Report:
    """Build Report from mind map state (drafts as section content)."""
    sections: list[Section] = []
    for leaf in mm.root.leaves():
        content = leaf.draft or f"**Claim:** {leaf.claim}"
        sections.append(Section(title=leaf.name, content=content, level=2))

    all_sources: list[Source] = []
    seen: set[str] = set()
    for uid in sorted(mm.snippets):
        src = mm.snippets[uid].source
        norm = normalize_url(src.url)
        if norm not in seen:
            seen.add(norm)
            all_sources.append(src)

    return Report(topic=topic, sections=sections, sources=all_sources)


# ── Writer: research drafts → human-like article ────────────────────

def _build_source_index(mm: MindMap) -> tuple[list[Source], dict[int, int]]:
    """Build global source list and snippet_uuid → 1-based global number mapping."""
    sources: list[Source] = []
    seen: set[str] = set()
    uuid_to_global: dict[int, int] = {}

    for uid in sorted(mm.snippets):
        src = mm.snippets[uid].source
        norm = normalize_url(src.url)
        if norm not in seen:
            seen.add(norm)
            sources.append(src)
        idx = next(i for i, s in enumerate(sources)
                   if normalize_url(s.url) == norm)
        uuid_to_global[uid] = idx + 1

    return sources, uuid_to_global


def _section_sources_text(leaf: MindMapNode, mm: MindMap,
                          uuid_to_global: dict[int, int]) -> str:
    """Format sources for a section with global numbering."""
    lines: list[str] = []
    seen: set[int] = set()
    for uid in sorted(leaf.snippet_uuids):
        gid = uuid_to_global.get(uid)
        if gid and gid not in seen and uid in mm.snippets:
            seen.add(gid)
            src = mm.snippets[uid].source
            lines.append(f"[{gid}] {src.title}\n{src.content[:2000]}")
    return "\n\n".join(lines)


def _remap_citations(text: str, leaf: MindMapNode,
                     mm: MindMap, uuid_to_global: dict[int, int]) -> str:
    """Replace local draft citations [1],[2]... with global numbers."""
    # Build local→global map for this section's snippets
    local_uids = sorted(uid for uid in leaf.snippet_uuids if uid in mm.snippets)
    local_to_global: dict[int, int] = {}
    for local_idx, uid in enumerate(local_uids, 1):
        gid = uuid_to_global.get(uid)
        if gid:
            local_to_global[local_idx] = gid

    if not local_to_global:
        return text

    def _replace(m):
        n = int(m.group(1))
        return f"[{local_to_global.get(n, n)}]"

    return re.sub(r'\[(\d+)\]', _replace, text)


def _tree_to_sections(nodes: list[MindMapNode],
                      leaf_texts: dict[int, str],
                      base_level: int = 2) -> list[Section]:
    """Convert tree nodes to Section list with proper header levels."""
    result: list[Section] = []
    for node in nodes:
        if node.children:
            result.append(Section(title=node.name, content="", level=base_level))
            result.extend(_tree_to_sections(node.children, leaf_texts, base_level + 1))
        else:
            text = leaf_texts.get(id(node), "")
            result.append(Section(title=node.name, content=text, level=base_level))
    return result


def _write_article(topic: str, mm: MindMap, st: dict, output_lang: str,
                   target_words: int, llm: LLM, ov: dict | None,
                   log: Callable) -> tuple[Report, dict]:
    """Transform research drafts into a polished article."""
    log("Writing article...")

    # Phase 1: PREPARE
    sources, uuid_to_global = _build_source_index(mm)
    leaves = mm.root.leaves()
    if not leaves:
        return Report(topic=topic, sections=[], sources=sources), {}

    # Enrich top sources with full page content
    n_enriched = enrich_sources(sources)
    if n_enriched:
        log(f"  Enriched {n_enriched} sources with full page content")

    anti_patterns = _get_prompt("ANTI_PATTERNS", ov)
    prose_techniques = _get_prompt("PROSE_TECHNIQUES", ov)
    style_techniques = st.get("techniques", "")
    style_tone = st.get("tone", "")

    # Word budget
    body_budget = int(target_words * 0.80)
    intro_budget = int(target_words * 0.10)
    concl_budget = int(target_words * 0.10)

    total_src = sum(len(leaf.snippet_uuids) for leaf in leaves)
    leaf_words: dict[str, int] = {}
    for leaf in leaves:
        weight = len(leaf.snippet_uuids) / total_src if total_src else 1 / len(leaves)
        leaf_words[leaf.name] = max(200, int(body_budget * weight))

    log(f"  {len(sources)} global sources, {len(leaves)} sections, ~{target_words} words target")

    # Phase 2: NARRATIVE_PLAN — works with top-level sections (groups, not leaves)
    top_sections = list(mm.root.children)

    parts: list[str] = []
    for i, sec in enumerate(top_sections):
        sec_leaves = sec.leaves()
        claim = sec.claim or (sec_leaves[0].claim if sec_leaves else "(none)")
        draft_preview = (sec_leaves[0].draft or "")[:300] if sec_leaves else ""
        line = f"{i}. [{sec.name}] claim: {claim}"
        if sec.children:
            line += "\n   subsections: " + ", ".join(c.name for c in sec.children)
        line += f"\n   draft preview: {draft_preview}"
        parts.append(line)
    sections_summary = "\n".join(parts)
    section_indices = ", ".join(str(i) for i in range(len(top_sections)))

    raw = llm(_get_prompt("NARRATIVE_PLAN", ov).format(
        topic=topic, sections_summary=sections_summary,
        section_indices=section_indices,
    ), max_tokens=1024, response_format={"type": "json_object"})
    plan = _parse_json(raw, expect="object")

    narrative_thread = plan.get("narrative_thread", topic)
    intro_hook = plan.get("intro_hook", "")
    conclusion_angle = plan.get("conclusion_angle", "")
    section_order = plan.get("section_order")

    # Reorder top-level sections (moves entire groups, not individual leaves)
    if isinstance(section_order, list) and len(section_order) == len(top_sections):
        try:
            top_sections = [top_sections[i] for i in section_order]
        except (IndexError, TypeError):
            pass

    # Derive ordered leaves from reordered top-level sections
    leaves = []
    for sec in top_sections:
        leaves.extend(sec.leaves())

    log(f"  Narrative: {narrative_thread[:100]}")

    # Phase 3: WRITE_SECTION (sequential for flow)
    section_texts: list[tuple[MindMapNode, str]] = []
    prev_ending = ""
    covered_topics: list[str] = []

    for i, leaf in enumerate(leaves):
        log(f"  Writing [{leaf.name}] ({i+1}/{len(leaves)})...")

        draft = leaf.draft or leaf.claim or ""
        draft_remapped = _remap_citations(draft, leaf, mm, uuid_to_global)
        src_text = _section_sources_text(leaf, mm, uuid_to_global)

        if prev_ending:
            prev_context = (
                f"PREVIOUS SECTION ENDING (continue the flow naturally):\n{prev_ending}"
            )
        else:
            prev_context = "This is the first body section."

        if covered_topics:
            prev_context += (
                "\n\nALREADY COVERED in previous sections (do NOT repeat — "
                "reference briefly if needed, but bring NEW angles):\n"
                + "\n".join(f"- {t}" for t in covered_topics)
            )

        text = llm(_get_prompt("WRITE_SECTION", ov).format(
            narrative_thread=narrative_thread,
            style_techniques=style_techniques,
            style_tone=style_tone,
            anti_patterns=anti_patterns,
            prose_techniques=prose_techniques,
            section_name=leaf.name,
            section_claim=leaf.claim or "",
            draft=draft_remapped,
            sources=src_text or "(no sources)",
            prev_section_context=prev_context,
            target_words=leaf_words.get(leaf.name, 400),
            output_lang_instruction=output_lang,
        ), max_tokens=4096)

        section_texts.append((leaf, text.strip()))
        prev_ending = text.strip()[-300:]
        covered_topics.append(f"[{leaf.name}]: {leaf.claim or leaf.name}")

    # Phase 4: WRITE_INTRO + WRITE_CONCLUSION
    first_beginning = section_texts[0][1][:300] if section_texts else ""

    log("  Writing intro...")
    intro = llm(_get_prompt("WRITE_INTRO", ov).format(
        topic=topic,
        narrative_thread=narrative_thread,
        intro_hook=intro_hook,
        style_techniques=style_techniques,
        style_tone=style_tone,
        anti_patterns=anti_patterns,
        prose_techniques=prose_techniques,
        first_section_beginning=first_beginning,
        target_words=intro_budget,
        output_lang_instruction=output_lang,
    ), max_tokens=2048).strip()

    # Trim intro if it bleeds into the first section
    if section_texts:
        first_text = section_texts[0][1]
        for overlap_len in range(min(len(intro), len(first_text), 200), 20, -1):
            if intro.endswith(first_text[:overlap_len]):
                intro = intro[:-overlap_len].rstrip()
                break

    log("  Writing conclusion...")
    section_claims = "\n".join(
        f"- {leaf.name}: {leaf.claim}" for leaf in leaves if leaf.claim)
    conclusion = llm(_get_prompt("WRITE_CONCLUSION", ov).format(
        topic=topic,
        narrative_thread=narrative_thread,
        conclusion_angle=conclusion_angle,
        section_claims=section_claims,
        style_techniques=style_techniques,
        style_tone=style_tone,
        anti_patterns=anti_patterns,
        prose_techniques=prose_techniques,
        target_words=concl_budget,
        output_lang_instruction=output_lang,
    ), max_tokens=2048).strip()

    # Phase 5: ASSEMBLE — preserve tree hierarchy in header levels
    leaf_texts = {id(leaf): text for leaf, text in section_texts}

    sections: list[Section] = []
    sections.append(Section(title="", content=intro, level=2))
    sections.extend(_tree_to_sections(top_sections, leaf_texts, base_level=2))
    sections.append(Section(title="", content=conclusion, level=2))

    word_count = sum(len(s.content.split()) for s in sections)
    log(f"  Article: {word_count} words, {len(sections)} sections, {len(sources)} sources")

    write_meta = {
        "narrative_thread": narrative_thread,
        "leaf_words": leaf_words,
        "uuid_to_global": uuid_to_global,
    }
    return Report(topic=topic, sections=sections, sources=sources), write_meta


# ── Review & Patch ────────────────────────────────────────────────────────

def _review_and_patch(report: Report, mm: MindMap, st: dict,
                      narrative_thread: str, output_lang: str,
                      leaf_words: dict[str, int],
                      uuid_to_global: dict[int, int],
                      llm_review: LLM, llm: LLM,
                      ov: dict | None, log: Callable) -> Report:
    """Review article with one model, patch flagged sections with another."""
    anti_patterns = _get_prompt("ANTI_PATTERNS", ov)
    prose_techniques = _get_prompt("PROSE_TECHNIQUES", ov)

    article_text = "\n\n".join(
        (f"## {s.title}\n{s.content}" if s.title else s.content)
        for s in report.sections if s.content)

    log("  Reviewing article...")
    raw = llm_review(_get_prompt("REVIEW_ARTICLE", ov).format(
        topic=report.topic,
        style_tone=st.get("tone", ""),
        style_techniques=st.get("techniques", ""),
        anti_patterns=anti_patterns,
        article_text=article_text,
        output_lang_instruction=output_lang,
    ), response_format={"type": "json_object"})
    review = _parse_json(raw, expect="object").get("sections", [])

    # Build critique lookup: title → critique text
    critiques: dict[str, str] = {}
    cuts: set[str] = set()
    for item in review:
        title = item.get("title", "")
        action = item.get("action", "ok")
        if action == "rewrite" and item.get("critique"):
            critiques[title.lower().strip()] = item["critique"]
        elif action == "cut":
            cuts.add(title.lower().strip())

    if not critiques and not cuts:
        log("  Review: all sections OK")
        return report

    log(f"  Review: {len(critiques)} to rewrite, {len(cuts)} to cut")

    # Cut sections
    if cuts:
        report.sections = [s for s in report.sections
                           if s.title.lower().strip() not in cuts]

    # Patch sections that need rewriting
    patched = 0
    prev_ending = ""
    for section in report.sections:
        key = section.title.lower().strip()
        if key not in critiques:
            if section.content:
                prev_ending = section.content[-300:]
            continue

        log(f"  Patching [{section.title}]...")

        # Find matching leaf for sources
        leaves = mm.root.leaves()
        leaf = next((l for l in leaves if l.name.lower().strip() == key), None)
        src_text = _section_sources_text(leaf, mm, uuid_to_global) if leaf else ""

        prev_context = (f"PREVIOUS SECTION ENDING:\n{prev_ending}"
                        if prev_ending else "This is the first body section.")

        text = llm(_get_prompt("PATCH_SECTION", ov).format(
            narrative_thread=narrative_thread,
            style_techniques=st.get("techniques", ""),
            style_tone=st.get("tone", ""),
            anti_patterns=anti_patterns,
            prose_techniques=prose_techniques,
            section_name=section.title,
            current_text=section.content,
            critique=critiques[key],
            sources=src_text or "(no sources)",
            prev_section_context=prev_context,
            target_words=leaf_words.get(section.title, 400),
            output_lang_instruction=output_lang,
        ), max_tokens=4096).strip()

        section.content = text
        prev_ending = text[-300:]
        patched += 1

    log(f"  Patched {patched} sections")
    return report


# ── Initialization ───────────────────────────────────────────────────────

def _decompose_and_search(topic: str, engine: SearchEngine, search_lang: str,
                          prof: dict, llm: LLM, ov: dict | None,
                          log: Callable) -> tuple[str, list[Source]]:
    """Decompose topic into queries, search each, return findings summary + sources."""
    raw = llm(_get_prompt("DECOMPOSE_TOPIC", ov).format(
        topic=topic, search_lang_instruction=search_lang,
    ), response_format={"type": "json_object"})
    queries = _parse_json(raw, expect="object").get("queries", [])
    if not queries:
        queries = [topic]
    log(f"  {len(queries)} queries:")
    for q in queries:
        log(f"    • {q}")

    all_sources: list[Source] = []
    seen_urls: set[str] = set()
    for q in queries:
        try:
            results = engine.search(q, max_results=prof["max_results"])
            for r in results:
                norm = normalize_url(r.url)
                if norm not in seen_urls:
                    seen_urls.add(norm)
                    all_sources.append(r)
        except Exception as e:
            log(f"    ⚠ search failed for «{q}»: {e}")

    all_sources = _filter_blocked_domains(all_sources)
    all_sources = _domain_diversify(all_sources, max_per_domain=3)
    all_sources = _filter_sources(all_sources, topic, llm, ov)
    log(f"  {len(all_sources)} unique sources after filtering")

    # Build findings text from source content
    findings_parts = []
    for i, src in enumerate(all_sources, 1):
        findings_parts.append(f"[{i}] {src.title}\n{src.content[:500]}")
    findings = "\n\n".join(findings_parts)

    return findings, all_sources


def _bootstrap(topic: str, findings: str, initial_sources: list[Source],
               st: dict, output_lang: str, prof: dict, llm: LLM,
               ov: dict | None, log: Callable) -> tuple[list[dict], MindMap]:
    """Generate experts + outline from findings. Returns (perspectives, MindMap)."""
    raw = llm(_get_prompt("BOOTSTRAP", ov).format(
        topic=topic, findings=findings,
        style_narrative=st["narrative"],
        n_perspectives=prof["n_perspectives"],
        output_lang_instruction=output_lang,
    ), response_format={"type": "json_object"})
    result = _parse_json(raw, expect="object")

    perspectives = result.get("perspectives", [])
    sections = result.get("sections", [])

    log(f"  {len(perspectives)} experts:")
    for p in perspectives:
        log(f"    • {p.get('role', '?')}: {p.get('description', '')[:80]}")

    mm = MindMap(topic)
    for s in sections:
        name = s.get("name", "")
        if not name:
            continue
        node = mm.add_child(mm.root, name)
        node.claim = s.get("claim", "")
        node.brief = s.get("brief", "")
        for ch in s.get("children", []):
            ch_name = ch.get("name", "")
            if ch_name:
                sub = mm.add_child(node, ch_name)
                sub.claim = ch.get("claim", "")

    log(f"  {len(mm.root.leaves())} sections:")
    for leaf in mm.root.leaves():
        claim_preview = f" — {leaf.claim[:70]}" if leaf.claim else ""
        log(f"    [{leaf.name}]{claim_preview}")

    # Seed reservoir with initial sources
    mm.add_to_reservoir(initial_sources)
    log(f"  {len(initial_sources)} sources → reservoir")

    return perspectives, mm


# ── Main ─────────────────────────────────────────────────────────────────

def research(
    topic: str,
    model: str | dict[str, str] = "anthropic/claude-sonnet-4-20250514",
    embedding_model: str = "text-embedding-3-small",
    api_base: str | None = None,
    search: str | SearchEngine = "duckduckgo",
    search_api_key: str | None = None,
    profile: str = "balanced",
    style: str = "analytical",
    search_languages: str | list[str] = "auto",
    output_language: str = "auto",
    target_words: int | None = None,
    prompts_override: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
    save_trace: bool = True,
) -> Report:
    """Research via roundtable discourse → polished article."""
    log = on_progress or (lambda _: None)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    trace = [] if save_trace else None
    _lock = threading.Lock()
    models = _resolve_models(model)
    llm = _make_llm(models["default"], api_base, usage, _lock, trace)
    llm_fast = _make_llm(models["fast"], api_base, usage, _lock, trace)
    llm_review = _make_llm(models["review"], api_base, usage, _lock, trace)

    prof = PROFILES.get(profile, PROFILES["balanced"])
    st = STYLES.get(style, STYLES["analytical"])
    tw = target_words or prof["target_words"]
    encoder = Encoder(embedding_model)

    if search_languages == "auto":
        search_lang = "Generate queries in the same language as the topic."
    elif isinstance(search_languages, list):
        search_lang = f"Generate queries in: {', '.join(search_languages)}."
    else:
        search_lang = f"Generate queries in {search_languages}."
    output_lang = (f"Write in {output_language}." if output_language != "auto"
                   else "Write in the same language as the topic.")

    engine = get_search_engine(search, search_api_key) if isinstance(search, str) else search
    ov = prompts_override

    try:
        # Step 1: Decompose topic → search queries → initial findings
        log("Exploring topic...")
        findings, initial_sources = _decompose_and_search(
            topic, engine, search_lang, prof, llm_fast, ov, log)

        # Step 2: Bootstrap experts + outline from findings
        log("Bootstrapping roundtable...")
        perspectives, mm = _bootstrap(
            topic, findings, initial_sources, st, output_lang, prof, llm, ov, log)

        # Step 3: Roundtable research
        _roundtable(topic, perspectives, mm, prof, search_lang, output_lang,
                    engine, encoder, llm, llm_fast, ov, log)

        # Write polished article
        report, write_meta = _write_article(topic, mm, st, output_lang, tw, llm, ov, log)

        # Review → Patch cycles (cross-model critique)
        review_cycles = prof.get("review_cycles", 0)
        for cycle in range(review_cycles):
            log(f"Review cycle {cycle + 1}/{review_cycles}...")
            report = _review_and_patch(
                report, mm, st, write_meta.get("narrative_thread", topic),
                output_lang, write_meta.get("leaf_words", {}),
                write_meta.get("uuid_to_global", {}),
                llm_review, llm, ov, log)

    except KeyboardInterrupt:
        log("\nInterrupted — returning partial results...")
        if 'mm' in locals():
            report = _render_debug(topic, mm)
        else:
            report = Report(topic=topic, sections=[], sources=[])

    report.trace = trace or []
    total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    if total_tokens:
        log(f"\nTokens: {total_tokens:,} (prompt: {usage['prompt_tokens']:,}, "
            f"completion: {usage['completion_tokens']:,})")
    log("\nDone!")
    return report
