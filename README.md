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
  material_considerations.json
  planning_balance.md
  planning_balance.json
  mc_conflicts.json
  evidence/
  RUN.yaml
  package.zip  (after `tpa package`)
```

`RUN.yaml` captures models used, retrieval knobs, config hash, and output file mapping. `reasoning.json`
records the multi-stage drafting diagnostics (policy framework summary, CEF table, critic issues), as well as claim→policy matches, conflicts, and the retrieval trace. Each run also emits `material_considerations.json`, `planning_balance.*`, and `mc_conflicts.json`, which respectively capture the per-topic Material Consideration assessments, the planning balance accumulator output, and any trade-offs the conflict resolver spotted. When `output.save_zip_of_pages` is true, page snapshots are exported and included both on disk and inside `package.zip` for traceability.

## Material Considerations & Planning Balance

- **Detection & Assessment** – chunks retain source-type metadata during ingestion, while `tpa.topics.TopicDetector` dynamically proposes material consideration topics by either querying the configured LLM or falling back to corpus-derived key phrases. `tpa.material_considerations` consumes the resulting topic map, auto-detects active topics per section, re-runs stratified retrieval plans, and produces structured findings (supporting/contradicting citations plus conclusions) even for previously unseen considerations.
- **Planner artefacts** – the MC assessments are written to `material_considerations.json` (with the raw retrieval trace), while the planning balance accumulator (`tpa.balance`) summarises harms/benefits/neutral points into both Markdown and JSON for downstream packaging.
- **Planning balance narrative** – `build_planning_balance` now calls back into the LLM (when available) to craft a professional narrative that satisficingly weighs harms and benefits; when offline, a hedged fallback paragraph is produced so officers never see a deterministic score-sheet. The Markdown log still lists citations per topic for traceability.
- **Conflict resolver** – `mc_conflicts.json` lists topic pairs where one MC recommends refusal and another recommends approval, so officers can surface explicit trade-offs in committee reports.
- **Section agent integration** – the multi-stage author now logs policy framework extracts, CEF tables, critic issues, repairs, and planning balance signals inside `reasoning.json`, ensuring verification and officer review can trace every conclusion.

### Style and Creativity (LLM-first)

- The text LLM defaults to `gpt-oss:20b` (Ollama) and drafts 3–5 stylistic variants per section (“policy-led”, “consultee-led”, “applicant-sceptical”), seeded with local policy phrasing (“plannerese”).
- A judge agent prefers paragraphs that: (i) cite policy, (ii) maintain officer hedging, (iii) align with the configured orientation (default `pro_delivery`). It then randomly selects among the best to preserve variety while staying policy‑led.
- Quantitative matters are extracted and listed strictly with citations under “Quantitative Matters”.

Configure in YAML under `style`:

```yaml
style:
  orientation: pro_delivery   # pro_delivery | balanced | neutral
  num_drafts: 3               # number of stylistic variants
  persona_variants: [policy_led, consultee_led, applicant_skeptical]
  combine_mode: judge_random  # or best_of
```

## Configuration (`CONFIG.sample.yml`)

```yaml
index:
  input_dirs:
    - ./tests/data/app
    - ./tests/data/policy
  ocr_fallback: false
retrieval:
  k_app: 4
  k_policy: 6
  k_adversarial: 2
  max_candidates: 80
  use_reranker: false
  mix_weights:
    applicant_case: 0.45
    policy: 0.4
    technical_report: 0.35
    consultee: 0.3
    objection: 0.25
    visual: 0.2
  recipes:
    default:
      applicant_case: {k: 4, weight: 0.3}
      policy: {k: 6, weight: 0.35}
      technical_report: {k: 3, weight: 0.2}
      consultee: {k: 2, weight: 0.1}
      objection: {k: 2, weight: 0.05}
llm:
  provider: dummy
  model: gpt-oss:20b
output:
  cite_style: inline_ids
  save_zip_of_pages: true
```

Set `llm.provider` to `ollama` to invoke a local `gpt-oss:20b` model (recommended for real runs). For automated tests or offline smoke checks you can set `provider=dummy` to skip the heavy call, but production runs should use Ollama. Embeddings now default to `qwen3-embedding:8b` via Ollama’s embeddings API; if the model is not running the pipeline raises a clear error (set `TPA_EMBED_MODE=hash` when you explicitly want the lightweight hash encoder). Visual chunks are summarised by `qwen3-vl:30b` so subjective matters (design quality, diagrams, maps) appear as `[VIS:...]` evidence — set `TPA_VISION_MODEL` to override or `TPA_DISABLE_VISION=1` to skip. Enabling `index.ocr_fallback: true` will attempt OCR (Tesseract + Pillow) when PyMuPDF returns empty text so scanned PDFs aren’t dropped. The `retrieval.recipes` block lets you stratify by source type (applicant case vs. policy vs. consultee), and default recipes already enforce the transport/heritage mixes used by the Material Considerations engine.

### Using Google Gemini 2.5 Pro (optional cloud backend)

You can switch to Google’s Gemini 2.5 Pro for both text drafting and multimodal (image) understanding:

1. Install deps (already in `pyproject.toml`): `google-genai`, `tenacity`.
2. Export your API key:
  ```bash
  export GOOGLE_API_KEY=your_key_here
  # or: export GEMINI_API_KEY=your_key_here
  ```
3. In your config:
  ```yaml
  llm:
    provider: google
    model: gemini-2.5-pro   # auto-normalised to models/gemini-2.5-pro
  ```
4. (Optional) Set `TPA_LLM_PROVIDER=google` to route vision summaries through Gemini instead of local Ollama.

When `provider: google` is active the pipeline uses the Gemini SDK (`google-genai`) to call the `gemini-2.5-pro` model. Vision chunks (page snapshots) are summarised via Gemini’s multimodal endpoint and injected as additional evidence lines (tagged `[VIS:...]`) so the main completion can reason over visual context. If the SDK or key are missing the run raises a clear error (or vision falls back to local model if only vision path fails). Set `TPA_DISABLE_VISION=1` to skip image analysis.

Future: streaming and long-context (1M token) support can be added by swapping to the SDK stream API; comments in `GoogleGeminiClient` show the extension point.

## Tests

Pytest suites cover both the end-to-end flow and the new architecture slices:

- `tests/test_smoke.py` – runs CLI end-to-end on the sample PDFs using `DummyEcho`.
- `tests/test_verify.py` – asserts each planner-specific check fails or passes appropriately.
- `tests/test_embed.py`, `tests/test_retrieve.py`, `tests/test_multimodal.py`, `tests/test_author.py`, `tests/test_material_considerations.py`, `tests/test_package.py` – unit tests for embeddings, stratified retrieval, vision summaries, the recursive author, MC assessments, and packaging.

Run everything with:

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

- Retrieval now enforces source mixing, adversarial samples, and per-section recipes; tune `retrieval.recipes` to add bespoke mixes for new sections.
- OCR fallback relies on `pytesseract` and `Pillow`; install them before enabling `index.ocr_fallback`.
- Multimodal summaries rely on `qwen3-vl:30b`; set `TPA_VISION_MODEL` or `TPA_DISABLE_VISION=1` to override during tests or offline runs.
- Evidence packaging exports the expanded artefact set (MC assessments, planning balance JSON/MD, conflicts) alongside page-level PNGs for traceability.

## Assumptions

- A local Ollama instance exposes `qwen3-embedding:8b` and `qwen3-vl:30b`. Set `TPA_EMBED_MODE=hash` only when you explicitly want the lightweight encoder for CI.
- Source-type tagging still relies on lightweight path/text cues inside `tpa.pdf_ingest`; extend `_infer_source_type` if your corpus uses very different naming conventions. Topic detection itself is handled dynamically via `TopicDetector`, which can call the configured LLM (or a local fallback) and therefore adapts to entirely new material considerations without code changes.
- Material Consideration activation maps sections → topics via `_SECTION_TOPIC_MAP`; extend that map when you add new prompt/section identifiers.
- The Dummy LLM provider is intended purely for tests; planner-grade runs should use an Ollama or Gemini backend that can handle the multi-stage drafting prompts.
