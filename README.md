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
- `tpa report --section transport --config CONFIG.yml --run RUN_NAME` – retrieves mixed policy/application evidence, drafts deterministic Markdown with inline IDs, and writes all artefacts under `runs/RUN_NAME/`.
- `tpa verify --run RUN_NAME` – runs planner-specific verification checks (policy linkage, source diversity, claim/evidence pairing, mini balance, file hygiene). Exits with status 1 on failure.
- `tpa package --run RUN_NAME` – zips the run output and evidence snapshots into `runs/RUN_NAME/package.zip`.

Each report run produces:

```
runs/<RUN_NAME>/
  section_<section>.md
  retrieved.json
  prompt.txt
  completion.md
  evidence/
  RUN.yaml
  package.zip  (after `tpa package`)
```

`RUN.yaml` captures models used, retrieval knobs, config hash, and output file mapping.

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

Set `llm.provider` to `ollama` to invoke a local `gpt-oss:20b` model. The current scaffold keeps authoring deterministic when `provider=dummy`, ensuring tests run offline. Embeddings default to `BAAI/bge-large-en`, with a hash-based fallback if the model is unavailable.

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

- Retrieval currently uses a lightweight hash embedding fallback if SentenceTransformers cannot load `BAAI/bge-large-en` (e.g., offline environments). Swap back to the true model once available.
- Reranker integration and full LLM drafting loops are stubbed but ready for extension (`tpa/rerank.py`, `tpa/llm.py`).
- Evidence packaging exports page-level PNGs for traceability; switch to PDF slices if preferred.
