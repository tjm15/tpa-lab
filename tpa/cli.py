from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from .author import PromptTemplate, compose_output, save_prompt
from .chunk_store import write_chunks
from .config import Config, load_config
from .logging import log
from .pdf_ingest import ingest_pdfs
from .embed import build_indexes
from .retrieve import retrieve
from .rerank import rerank
from .verify import verify_run
from .package import package_run

app = typer.Typer(help="Decision-led officer report pipeline")

INDEX_ROOT = Path("runs/index")
CHUNKS_PATH = INDEX_ROOT / "chunks.jsonl"
INDEX_DIR = INDEX_ROOT / "indexes"


def _ensure_index(config: Config) -> None:
    if CHUNKS_PATH.exists() and INDEX_DIR.exists():
        return
    chunks = ingest_pdfs(config)
    write_chunks(chunks, CHUNKS_PATH)
    build_indexes(chunks, INDEX_DIR)


@app.command()
def index(config: Path = typer.Option(..., exists=True, readable=True, help="Config YAML path")) -> None:
    cfg = load_config(config)
    chunks = ingest_pdfs(cfg)
    log("ingest", status="completed", count=len(chunks))
    write_chunks(chunks, CHUNKS_PATH)
    build_indexes(chunks, INDEX_DIR)
    log("embed", status="completed", chunks_path=str(CHUNKS_PATH))


@app.command()
def report(
    section: str = typer.Option("transport", help="Section identifier (e.g., transport)"),
    config: Path = typer.Option(..., exists=True, readable=True, help="Config YAML path"),
    run: str = typer.Option(..., help="Run name for outputs"),
) -> None:
    cfg = load_config(config)
    _ensure_index(cfg)

    run_dir = Path("runs") / run
    run_dir.mkdir(parents=True, exist_ok=True)

    query = f"{section} policy compliance"
    retrieved = retrieve(query, CHUNKS_PATH, INDEX_DIR, cfg.retrieval)
    retrieved = rerank(retrieved, cfg.retrieval.use_reranker)

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

    markdown, prompt_text = compose_output(section, retrieved, template, llm=None)

    prompt_file = run_dir / "prompt.txt"
    save_prompt(prompt_file, prompt_text)
    completion_file = run_dir / "completion.md"
    completion_file.write_text(markdown, encoding="utf-8")
    output_file = run_dir / f"section_{section}.md"
    output_file.write_text(markdown, encoding="utf-8")

    if cfg.output.save_zip_of_pages:
        from .evidence import export_pages

        export_pages(retrieved, run_dir / "evidence")

    manifest = {
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "embedding": "BAAI/bge-large-en",
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


if __name__ == "__main__":  # pragma: no cover
    app()
