"""Interactive CLI for PORF."""

import argparse
import json
import sys
import time
from pathlib import Path

from . import research, PROFILES, STYLES

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import questionary
    from questionary import Style
    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False

STYLE = Style([
    ('qmark', 'fg:cyan bold'), ('question', 'fg:white bold'),
    ('answer', 'fg:cyan'), ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan'), ('selected', 'fg:cyan'),
]) if QUESTIONARY_AVAILABLE else None


def _ask(func, *args, **kwargs):
    """Ask a questionary question; exit on cancel."""
    result = func(*args, style=STYLE, **kwargs).ask()
    if result is None:
        print("Cancelled.")
        sys.exit(0)
    return result


def _parse_languages(lang: str):
    if lang == "auto":
        return "auto"
    if "," in lang:
        return [l.strip() for l in lang.split(",")]
    return lang


def wizard_mode(console):
    """Interactive wizard for research configuration."""
    if not QUESTIONARY_AVAILABLE:
        print("Interactive mode requires 'questionary'. Install: pip install porf[cli]")
        sys.exit(1)

    if console:
        console.print()
        console.print(Panel.fit("[bold cyan]PORF[/] — Publish Original Research Fast", border_style="cyan"))
        console.print()

    topic = _ask(questionary.text, "Research topic:")
    if not topic:
        print("Cancelled.")
        sys.exit(0)

    profile = _ask(questionary.select, "Profile:", choices=[
        questionary.Choice("quick    — Fast draft, ~3 min", value="quick"),
        questionary.Choice("balanced — Good for most topics, ~10 min", value="balanced"),
        questionary.Choice("deep     — Thorough research, ~20 min", value="deep"),
    ], default="balanced")

    style = _ask(questionary.select, "Style:", choices=[
        questionary.Choice("analytical   — Systematic analysis", value="analytical"),
        questionary.Choice("academic     — Formal academic review", value="academic"),
        questionary.Choice("journalistic — Investigative long-form", value="journalistic"),
        questionary.Choice("popular      — Accessible for general audience", value="popular"),
        questionary.Choice("essay        — Reflective, thoughtful essay", value="essay"),
    ], default="analytical")

    lang = _ask(questionary.select, "Search language:", choices=[
        questionary.Choice("Auto-detect from topic", value="auto"),
        questionary.Choice("English only", value="en"),
        questionary.Choice("Russian only", value="ru"),
        questionary.Choice("English + Russian", value="en,ru"),
        questionary.Choice("Custom...", value="custom"),
    ], default="auto")

    if lang == "custom":
        lang = _ask(questionary.text, "Languages (comma-separated, e.g. en,de,fr):") or "auto"

    languages = _parse_languages(lang)

    output_language = "auto"
    if isinstance(languages, list) and len(languages) > 1:
        choices = [questionary.Choice("Auto (same as topic)", value="auto")] + [
            questionary.Choice(l.upper(), value=l) for l in languages
        ]
        output_language = _ask(questionary.select, "Output language:", choices=choices, default="auto")
    elif isinstance(languages, str) and languages != "auto":
        output_language = languages  # single explicit language → use it for output too

    search_engine = _ask(questionary.select, "Search engine:", choices=[
        questionary.Choice("DuckDuckGo (free, no API key)", value="duckduckgo"),
        questionary.Choice("Tavily (best for research)", value="tavily"),
        questionary.Choice("Brave Search", value="brave"),
        questionary.Choice("Serper (Google)", value="serper"),
        questionary.Choice("SearxNG (self-hosted)", value="searxng"),
    ], default="duckduckgo")

    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
    default_fn = f"{safe[:50].strip().replace(' ', '_').lower()}.md" or "report.md"

    output = _ask(questionary.select, "Where to save the report?", choices=[
        questionary.Choice(f"Save to {default_fn}", value=default_fn),
        questionary.Choice("Save to custom path...", value="custom"),
        questionary.Choice("Print to terminal (don't save)", value="print"),
    ], default=default_fn)

    if output == "custom":
        output = _ask(questionary.text, "File path:", default=default_fn)

    model, api_base = "anthropic/claude-sonnet-4-20250514", None
    if _ask(questionary.confirm, "Use custom model? (default: anthropic/claude-sonnet-4-20250514)", default=False):
        model = _ask(questionary.text, "Model name (e.g. anthropic/claude-sonnet-4-20250514, anthropic/claude-3-5-sonnet):",
                     default="anthropic/claude-sonnet-4-20250514")
        api_base = questionary.text("API base URL (leave empty for default):", style=STYLE).ask() or None

    return {
        "topic": topic, "profile": profile, "style": style,
        "languages": languages, "output_language": output_language,
        "search": search_engine, "model": model, "api_base": api_base,
        "output": output,
    }


def _progress_log(console, msg: str):
    """Simple colorized progress output."""
    s = msg.strip()
    if not s or "Research complete" in s:
        return
    if not console:
        print(msg)
        return

    phases = ("Exploring", "Bootstrapping", "Roundtable", "Writing")
    if any(k in s for k in phases):
        console.print(f"\n[bold cyan]▸ {s}[/]")
    elif "sources" in s.lower() or "done" in s.lower():
        console.print(f"  [green]✓ {s}[/]")
    elif "error" in s.lower() or "failed" in s.lower():
        console.print(f"  [yellow]⚠ {s}[/]")
    elif s.startswith("Tokens"):
        console.print(f"\n[dim]{s}[/]")
    else:
        console.print(f"  [dim]{s}[/]")


def run_research(config, console):
    """Run research with progress display."""
    start = time.time()
    try:
        report = research(
            topic=config["topic"], profile=config["profile"],
            style=config.get("style", "analytical"),
            model=config["model"], api_base=config.get("api_base"),
            search=config.get("search", "duckduckgo"),
            search_languages=config["languages"],
            output_language=config.get("output_language", "auto"),
            target_words=config.get("target_words"),
            on_progress=lambda msg: _progress_log(console, msg),
            save_trace=config.get("save_trace", True),
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

    mins, secs = divmod(int(time.time() - start), 60)
    if console:
        console.print()
        console.print(Panel(
            f"[green]✓[/] [bold]{len(report.sources)}[/] sources · "
            f"[bold]{len(report.sections)}[/] sections · "
            f"[bold]{mins}m {secs}s[/]",
            title="[bold green]Done![/]", border_style="green",
        ))
    else:
        print(f"\nDone! {len(report.sources)} sources, {len(report.sections)} sections, {mins}m {secs}s")
    return report


def main():
    parser = argparse.ArgumentParser(description="PORF — Publish Original Research Fast")
    parser.add_argument("topic", nargs="?", help="Research topic")
    parser.add_argument("-p", "--profile", choices=list(PROFILES.keys()), default="balanced")
    parser.add_argument("-t", "--style", choices=list(STYLES.keys()), default="analytical")
    parser.add_argument("-m", "--model", default="anthropic/claude-sonnet-4-20250514")
    parser.add_argument("--model-fast", help="Fast/cheap model for queries and filtering")
    parser.add_argument("--model-review", help="Strong model for review phase")
    parser.add_argument("-s", "--search", default="duckduckgo",
                        choices=["duckduckgo", "tavily", "brave", "serper", "searxng"])
    parser.add_argument("-l", "--lang", default="auto", help="Search language(s): auto, en, ru, or en,ru")
    parser.add_argument("--out-lang", default="auto", help="Output language: auto, en, ru, etc.")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-w", "--words", type=int, default=None, help="Target article length in words")
    parser.add_argument("--api-base", help="API base URL for local models")
    parser.add_argument("--no-trace", action="store_true", help="Disable LLM call trace logging")

    args = parser.parse_args()
    console = Console() if RICH_AVAILABLE else None

    if not args.topic:
        if not RICH_AVAILABLE or not QUESTIONARY_AVAILABLE:
            print("Interactive mode requires rich and questionary.")
            print("Install: pip install porf[cli]")
            print("\nOr: porf \"Your topic\"")
            sys.exit(1)
        config = wizard_mode(console)
    else:
        model = args.model
        if args.model_fast or args.model_review:
            model = {"default": args.model,
                     "fast": args.model_fast or args.model,
                     "review": args.model_review or args.model}
        config = {
            "topic": args.topic, "profile": args.profile, "style": args.style,
            "languages": _parse_languages(args.lang), "output_language": args.out_lang,
            "search": args.search, "model": model,
            "api_base": args.api_base, "target_words": args.words,
            "save_trace": not args.no_trace,
        }

    if console:
        console.print()
        console.print(Panel.fit(f"[bold]{config['topic']}[/]", title="[bold cyan]PORF[/]", border_style="cyan"))
        m = config['model']
        model_str = m if isinstance(m, str) else f"{m.get('default', '?')} (fast={m.get('fast', '-')}, review={m.get('review', '-')})"
        info = f"Profile: {config['profile']} | Style: {config.get('style', 'analytical')} | Model: {model_str}"
        console.print(f"[dim]{info}[/]\n")

    report = run_research(config, console)

    output = args.output or config.get("output")
    if output and output != "print":
        Path(output).write_text(report.markdown)
        if console:
            console.print(f"\n[green]✓[/] Saved to [bold]{output}[/]")
    else:
        print(report.markdown)

    # Save trace
    if report.trace:
        if output and output != "print":
            trace_path = str(Path(output).with_suffix(".trace.jsonl"))
        else:
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in config["topic"])
            trace_path = f"{safe[:50].strip().replace(' ', '_').lower() or 'porf'}.trace.jsonl"
        with open(trace_path, "w") as f:
            for entry in report.trace:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if console:
            console.print(f"[green]✓[/] Trace: [bold]{trace_path}[/] ({len(report.trace)} LLM calls)")


if __name__ == "__main__":
    main()
