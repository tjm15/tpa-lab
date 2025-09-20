from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from .author import PromptTemplate, compose_output, save_prompt
from .chunk_store import write_chunks
from .config import Config, load_config
from .logging import log
from .pdf_ingest import ingest_pdfs
from .datasets import discover_corpora
from .embed import build_indexes
from .retrieve import retrieve
from .multimodal import summarise_visuals
from .rerank import rerank
from .verify import verify_run
from .package import package_run
from .reason import build_reasoning
from .llm import get_llm

app = typer.Typer(help="Decision-led officer report pipeline")

INDEX_ROOT = Path("runs/index")
CHUNKS_PATH = INDEX_ROOT / "chunks.jsonl"
INDEX_DIR = INDEX_ROOT / "indexes"


def _ensure_index(config: Config, show_progress: bool = False) -> None:
    if CHUNKS_PATH.exists() and INDEX_DIR.exists():
        return
    chunks = ingest_pdfs(config, show_progress=show_progress)
    write_chunks(chunks, CHUNKS_PATH)
    build_indexes(chunks, INDEX_DIR)


@app.command()
def index(
    config: Path = typer.Option(..., exists=True, readable=True, help="Config YAML path"),
    auto_discover: bool = typer.Option(
        False,
        help="If set, ignore index.input_dirs in config and auto-discover 'downloads/policies' and latest 'downloads/earls-court-*' directory.",
    ),
    progress: bool = typer.Option(
        True, help="Show a progress bar during PDF ingestion (default on)."
    ),
) -> None:
    cfg = load_config(config)
    if auto_discover:
        corpora = discover_corpora()
        new_dirs = corpora.existing
        if not new_dirs:
            log("discover", status="empty", message="No corpora found under downloads/")
        else:
            cfg.index.input_dirs = new_dirs
            log(
                "discover",
                status="ok",
                input_dirs=[str(p) for p in new_dirs],
            )
    chunks = ingest_pdfs(cfg, show_progress=progress)
    log("ingest", status="completed", count=len(chunks))
    write_chunks(chunks, CHUNKS_PATH)
    build_indexes(chunks, INDEX_DIR)
    log("embed", status="completed", chunks_path=str(CHUNKS_PATH))


@app.command()
def report(
    section: str = typer.Option("transport", help="Section identifier (e.g., transport)"),
    config: Path = typer.Option(..., exists=True, readable=True, help="Config YAML path"),
    run: str = typer.Option(..., help="Run name for outputs"),
    auto_discover: bool = typer.Option(
        False, help="Auto-discover corpora (policies + latest earls court) before running."
    ),
    progress: bool = typer.Option(True, help="Show progress bar if re-indexing is needed."),
) -> None:
    cfg = load_config(config)
    if auto_discover:
        corpora = discover_corpora()
        new_dirs = corpora.existing
        if new_dirs:
            cfg.index.input_dirs = new_dirs
            log("discover", status="ok", input_dirs=[str(p) for p in new_dirs])
        else:
            log("discover", status="empty")
    elif "TPA_AUTO_DISCOVER" in os.environ:  # backwards compatibility
        corpora = discover_corpora()
        new_dirs = corpora.existing
        if new_dirs:
            cfg.index.input_dirs = new_dirs
            log("discover", status="ok", input_dirs=[str(p) for p in new_dirs])
    _ensure_index(cfg, show_progress=progress)

    run_dir = Path("runs") / run
    run_dir.mkdir(parents=True, exist_ok=True)

    query = f"{section} policy compliance"
    retrieved = retrieve(query, CHUNKS_PATH, INDEX_DIR, cfg.retrieval)
    retrieved = rerank(retrieved, cfg.retrieval.use_reranker)
    if not retrieved:
        log("retrieve", status="empty", run=run, section=section)

    retrieved_payload = [
        {
            "id": item.chunk.id,
            "kind": item.chunk.kind,
            "score": item.score,
            "path": item.chunk.path,
            "page": item.chunk.page,
        }
        for item in retrieved
    ]
    (run_dir / "retrieved.json").write_text(json.dumps(retrieved_payload, indent=2), encoding="utf-8")

    template_path = Path(__file__).resolve().parent / "prompts" / f"{section}.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found for section '{section}'")
    template = PromptTemplate.from_yaml(template_path)

    llm = get_llm(cfg.llm.provider, cfg.llm.model)
    # Summarise visuals (visual chunks are already mixed in retrieval; we filter)
    visual_chunks = [r for r in retrieved if r.chunk.kind == "visual"]
    visual_summaries = summarise_visuals(visual_chunks)

    markdown, prompt_text, completion_text = compose_output(
        section, retrieved, template, llm=llm, visual_summaries=visual_summaries
    )

    prompt_file = run_dir / "prompt.txt"
    save_prompt(prompt_file, prompt_text)
    completion_file = run_dir / "completion.md"
    completion_file.write_text(completion_text if completion_text else markdown, encoding="utf-8")
    output_file = run_dir / f"section_{section}.md"
    output_file.write_text(markdown, encoding="utf-8")

    reasoning_path = build_reasoning(section, retrieved, run_dir)

    if cfg.output.save_zip_of_pages:
        from .evidence import export_pages

        export_pages(retrieved, run_dir / "evidence")

    manifest = {
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "embedding": "BAAI/bge-large-en-v1.5",
            "reranker": None,
            "llm": cfg.llm.model,
        },
        "inputs": {
            "sections": [section],
            "config_hash": cfg.hash,
        },
        "retrieval": {
            "k_app": cfg.retrieval.k_app,
            "k_policy": cfg.retrieval.k_policy,
            "k_adversarial": cfg.retrieval.k_adversarial,
            "mix_weights": cfg.retrieval.mix_weights,
        },
        "files": {
            "output_md": output_file.name,
            "retrieved": "retrieved.json",
            "prompt": prompt_file.name,
            "completion": completion_file.name,
            "reasoning": reasoning_path.name,
            "evidence_dir": "evidence" if cfg.output.save_zip_of_pages else None,
        },
        "output": {
            "cite_style": cfg.output.cite_style,
            "save_zip_of_pages": cfg.output.save_zip_of_pages,
        },
    }

    import yaml

    (run_dir / "RUN.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    log("report", run=run, section=section)


@app.command()
def verify(run: str = typer.Option(..., help="Run name")) -> None:
    run_dir = Path("runs") / run
    success = verify_run(run_dir)
    if not success:
        raise typer.Exit(code=1)


@app.command()
def package(run: str = typer.Option(..., help="Run name")) -> None:
    run_dir = Path("runs") / run
    package_path = package_run(run_dir)
    log("package", run=run, package=str(package_path))


@app.command()
def pipeline(
    run: str = typer.Option(..., help="Run name for outputs"),
    config: Path = typer.Option(..., exists=True, readable=True, help="Config YAML path"),
    sections: str = typer.Option(
        "transport",
        help="Comma-separated section identifiers. Use '*' to process all prompt templates in tpa/prompts/.",
    ),
    auto_discover: bool = typer.Option(
        True, help="Auto-discover corpora (policies + latest earls court) before indexing (default on)."
    ),
    progress: bool = typer.Option(
        True, help="Show a progress bar during PDF ingestion if index build required."
    ),
) -> None:
    """End-to-end: (auto-discover) -> index (if needed) -> report(s)."""
    cfg = load_config(config)
    if auto_discover:
        corpora = discover_corpora()
        new_dirs = corpora.existing
        if new_dirs:
            cfg.index.input_dirs = new_dirs
            log("discover", status="ok", input_dirs=[str(p) for p in new_dirs])
        else:
            log("discover", status="empty")

    # Ensure index exists for these dirs
    _ensure_index(cfg, show_progress=progress)

    # Resolve sections list
    resolved_sections = []
    if sections.strip() == "*":
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        for p in sorted(prompts_dir.glob("*.yaml")):
            resolved_sections.append(p.stem)
    else:
        for s in sections.split(","):
            s = s.strip()
            if s:
                resolved_sections.append(s)

    if not resolved_sections:
        log("pipeline", status="no_sections")
        return

    for sec in resolved_sections:
        typer.echo(f"[pipeline] generating section '{sec}'")
        # Call report logic without re-discovery to avoid resetting dirs mid-loop
        report(section=sec, config=config, run=run, auto_discover=False)

    log("pipeline", run=run, sections=resolved_sections)


if __name__ == "__main__":  # pragma: no cover
    app()
