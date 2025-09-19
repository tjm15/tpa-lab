# TPA Decision-Led Officer Pipeline

This repository provides a minimal, decision-led pipeline that turns PDF “soup” (application + policy) into planner-grade section outputs with inline citations, verification checks, and packaging utilities. The implementation follows the **Agent Implementation Spec (v1.0)** requirements outlined in `SPECS.md`.

## Quick Start

```bash
make setup                 # install project dependencies
make index CONFIG=CONFIG.sample.yml
make report SECTION=transport CONFIG=CONFIG.sample.yml RUN=dev
make verify RUN=dev        # planner-specific checks (fails with non-zero on issues)
make package RUN=dev       # bundle markdown, evidence, manifest
```

The sample configuration points at `tests/data/app` and `tests/data/policy`, which contain tiny PDFs suitable for smoke testing. Replace these input directories with your own corpus when running against real documents.

### CLI commands

- `tpa index --config CONFIG.yml` – parses PDFs into a chunk store (`runs/index/chunks.jsonl`) and builds embeddings per source class (`runs/index/indexes/`).
- `tpa report --section transport --config CONFIG.yml --run RUN_NAME` – retrieves mixed policy/application evidence, calls the configured local LLM (default `ollama:gpt-oss:20b`) to produce officer prose, and writes all artefacts (prompt, raw completion, curated section markdown, reasoning log, evidence snapshots) under `runs/RUN_NAME/`.
- `tpa verify --run RUN_NAME` – runs planner-specific verification checks (policy linkage, source diversity, claim/evidence pairing, mini balance, file hygiene). Exits with status 1 on failure.
- `tpa package --run RUN_NAME` – zips the run output and evidence snapshots into `runs/RUN_NAME/package.zip`.

Each report run produces:

```
runs/<RUN_NAME>/
  section_<section>.md
  retrieved.json
  prompt.txt
  completion.md
  reasoning.json
  evidence/
  RUN.yaml
  package.zip  (after `tpa package`)
```

`RUN.yaml` captures models used, retrieval knobs, config hash, and output file mapping. `reasoning.json`
records the claim→policy matches, conflicts, and retrieval trace used to draft the section. When `output.save_zip_of_pages` is true, page snapshots are exported and included both on disk and inside `package.zip` for traceability.

## Configuration (`CONFIG.sample.yml`)

```yaml
index:
  input_dirs:
    - ./tests/data/app
    - ./tests/data/policy
  ocr_fallback: false
retrieval:
  k_app: 3
  k_policy: 3
  k_adversarial: 0
  max_candidates: 60
  use_reranker: false
  mix_weights: {app: 0.5, policy: 0.5}
llm:
  provider: dummy
  model: gpt-oss:20b
output:
  cite_style: inline_ids
  save_zip_of_pages: true
```

Set `llm.provider` to `ollama` to invoke a local `gpt-oss:20b` model (recommended for real runs). For automated tests or offline smoke checks you can set `provider=dummy` to skip the heavy call, but production runs should use Ollama. Embeddings default to `BAAI/bge-large-en-v1.5`; if the model cannot be loaded, we fall back to a hash encoder and log the downgrade. Enabling `index.ocr_fallback: true` will attempt OCR (Tesseract + Pillow) when PyMuPDF returns empty text so scanned PDFs aren’t dropped. Visual chunks are handed to `gemma3:27b` via Ollama for multimodal commentary, so ensure that model is pulled locally.

## Tests

Two pytest suites ship with the repo:

- `tests/test_smoke.py` – runs CLI end-to-end on the sample PDFs using `DummyEcho`.
- `tests/test_verify.py` – asserts each planner-specific check fails or passes appropriately.

Run them with:

```bash
pytest
```

## Makefile Targets

```
make setup    # pip install -e .[test]
make index    # wrapper around tpa index
make report   # SECTION=<name> RUN=<id>
make verify   # RUN=<id>
make package  # RUN=<id>
```

## Notes & TODOs

- Retrieval now enforces source mixing and supports optional adversarial samples; set `k_adversarial` and `use_reranker` in the config to experiment with richer evidence blends.
- OCR fallback relies on `pytesseract` and `Pillow`; install them before enabling `index.ocr_fallback`.
- Multimodal summaries rely on `gemma3:27b` via Ollama; set `TPA_DISABLE_VISION=1` to skip vision calls during tests or offline runs.
- Evidence packaging exports page-level PNGs for traceability; switch to PDF slices if preferred.
