Implement to this outcomes spec. If any detail is ambiguous, choose the simplest option that satisfies the DoD and document the choice in README under 'Assumptions'.

# North Star

A **decision-led** pipeline that turns a messy PDF soup (application + policy) into a **planner-grade officer report** with traceable evidence, scrutiny of applicant claims, and explicit planning-balance reasoning.

---

## Agent Implementation Spec (v1.0)

**Audience:** a capable coding agent. **Outcome:** a running, end‑to‑end pipeline that produces planner‑grade section drafts with citations and passes planner‑specific verification checks — not just a toy demo.

### 1) Environment

* Python 3.11+. OS: Linux or Windows.
* GPU optional. Must run on CPU. Prefer PyTorch with CUDA if available.
* All models local by default. Provide adapters for:

  * **Embeddings:** `BAAI/bge-large-en` via `sentence-transformers`.
  * **Reranker (optional):** `bge-reranker-large` via `sentence-transformers` cross‑encoder.
  * **LLM:** abstract interface; implement `OllamaLocal(model="gpt-oss:20b")` and `DummyEcho` (for tests).

### 2) Project layout

```
repo/
  tpa/
    __init__.py
    cli.py
    config.py
    logging.py
    pdf_ingest.py
    chunk_store.py          # page chunks + metadata
    embed.py                # embeddings + index build
    retrieve.py             # mixing policy/app + adversarial
    rerank.py               # optional cross-encoder
    llm.py                  # BaseLLM, OllamaLocal, DummyEcho
    prompts/
      transport.yaml
      heritage.yaml
      design_townscape.yaml
    author.py               # CEF tables + drafting
    verify.py               # planner-specific checks
    package.py              # evidence zip + manifest
  tests/
    data/
      sample_app.pdf
      sample_policy.pdf
    test_smoke.py
    test_verify.py
  runs/                     # created at runtime
  pyproject.toml
  README.md
  Makefile
```

### 3) Config (YAML)

```yaml
index:
  input_dirs:
    - ./data/app
    - ./data/policy
  ocr_fallback: false
retrieval:
  k_app: 6
  k_policy: 6
  k_adversarial: 0
  max_candidates: 60
  use_reranker: false
  mix_weights: {app: 0.5, policy: 0.5}
llm:
  provider: ollama
  model: gpt-oss:20b
output:
  cite_style: inline_ids
  save_zip_of_pages: true
```

### 4) Prompts schema

```yaml
role: "You are a UK planning officer writing a section of an officer report."
structure:
  headings: ["Claims", "Evidence", "Finding", "Risks and Uncertainties", "Mini Planning Balance"]
style:
  tone: "professional, precise, sceptical of applicant claims"
  citation: "Inline source IDs like [APP:foo_p12] or [POL:lp_sp7_p34]"
constraints:
  - "Use only retrieved evidence; do not invent citations."
  - "Every paragraph must include at least one policy citation where applicable."
query_plan:
  - "List applicant claims relevant to the section."
  - "For each claim, fetch constraining policy/standards."
  - "Draft finding and mini planning balance."
```

### 5) CLI commands (must exist)

* `tpa index --config CONFIG.yml` → builds a JSONL chunk store with fields `{id, kind, path, page, text, hash}` and an embeddings index.
* `tpa report --section transport --config CONFIG.yml --run RUN_NAME` → retrieves, drafts Markdown, writes outputs.
* `tpa verify --run RUN_NAME` → runs planner‑specific checks (see §7) and prints a pass/fail summary.
* `tpa package --run RUN_NAME` → zips retrieved page PDFs and writes `RUN.yaml` manifest.

### 6) Outputs (must be created)

Create `runs/{RUN_NAME}/` containing:

* `section_transport.md` (or chosen section)
* `retrieved.json` (ordered list with `{id, kind, score, rerank_score?}`)
* `prompt.txt` (final composed prompt given to the LLM)
* `completion.md` (raw LLM output before post‑processing)
* `evidence/` (PDF page images or slices if `save_zip_of_pages=true`)
* `RUN.yaml` manifest with:

```yaml
run: RUN_NAME
created_at: <iso8601>
models:
  embedding: BAAI/bge-large-en
  reranker: null
  llm: gpt-oss:20b
inputs:
  sections: [transport]
  config_hash: <sha256>
retrieval:
  k_app: 6
  k_policy: 6
  k_adversarial: 0
  mix_weights: {app: 0.5, policy: 0.5}
files:
  output_md: section_transport.md
  retrieved: retrieved.json
  prompt: prompt.txt
  completion: completion.md
  evidence_dir: evidence/
```

### 7) Planner‑specific verification (no generic ML eval)

Implement `tpa.verify.checks` and fail the run if any check fails:

1. **Policy linkage:** every paragraph in the output contains ≥1 `[POL:*]` citation when the section is policy‑relevant.
2. **Source diversity:** output cites ≥3 distinct source IDs overall, with at least one policy and one application source.
3. **Claim–evidence pairing:** for each explicit applicant claim detected (look for quoted spans or “the applicant states”), there is adjacent evidence or counter‑evidence.
4. **Mini balance present:** a “Mini Planning Balance” subsection exists and is non‑empty.
5. **File hygiene:** all paths in `retrieved.json` resolve; `evidence/` exists if requested.

### 8) Retrieval rules

* Maintain **source separation**: build two indices (`kind = app|policy`).
* Mix results by `k_policy`/`k_app`; never allow one class to exceed 70% of final context.
* Optional: adversarial mode fetches low‑similarity but high‑conflict candidates (e.g., policies with thresholds that might contradict claims).
* Rerank (if enabled) only reorders the candidate pool; do not change the class mix.

### 9) Authoring rules

* Author from retrieved snippets only; include inline IDs.
* Produce **CEF tables** in Markdown if easily derivable; otherwise, keep prose with clearly marked Claims/Evidence/Findings.
* End with a **Mini Planning Balance** paragraph specific to the section.

### 10) Error handling & logging

* Structured logs to stdout (JSON) with event types: `ingest`, `embed`, `retrieve`, `rerank`, `draft`, `verify`, `package`.
* On exception, write a `RUN_ERROR.yaml` with stack trace and stage.
* CLI must exit non‑zero on verification failure.

### 11) Tests (pytest)

* `test_smoke.py`: end‑to‑end on tiny fixtures (two‑page app, two‑page policy) using `DummyEcho` to ensure plumbing works.
* `test_verify.py`: synthetic output files that violate each check; assert failures, then a passing case.

### 12) Make targets

```
make setup      # install deps
make index      # runs tpa index
make report     # runs tpa report SECTION=transport RUN=dev
make verify     # runs tpa verify RUN=dev
make package    # runs tpa package RUN=dev
```

### 13) Definition of Done

* `make report` produces Markdown with inline IDs; `make verify` returns success (0); `make package` creates a distributable folder with evidence and manifest.
* No unhandled exceptions on CPU‑only machines.

### 14) Non-goals (for now)

* Full multimodal parsing, GIS overlays, or design-code VLM checks.
* Fancy provenance UIs; JSON/YAML manifests are sufficient.

---

## 0) Minimal Flow (hour-one)

**Goal:** Have *something that runs* end-to-end.

* Input: one folder of PDFs (application + policy allowed).
* Chunking: naïve page chunks via PyMuPDF.
* Embeddings: `bge-large` (text-only) or ColPali (page-level) — pick one.
* Store: SQLite / JSONL index (paths, page, text, embedding vector id).
* Retrieval: top-k (k=10) by cosine.
* LLM: `gpt-oss:20b` (text-only) with a simple section prompt.
* Output: Markdown for one section (e.g., **Transport**), plus a JSON run record `{query, retrieved_ids, prompt, completion}`.
* CLI command: `tpa report --section transport --k 10 --run-name 2025-09-18T...`

**Deliverable:** A single report section that cites the IDs of retrieved pages.

---

## 0.5) Quick Wins (same day)

* **Section set:** transport, design/townscape, heritage, daylight/sunlight, ecology/BNG.
* **Source guardrails:** separate indices for **policy** vs **application**; allow `k_policy` + `k_app` with mixing ratio.
* **Rerank:** add a cross-encoder reranker (`bge-reranker-large`) to reorder the top 50 → top 10.
* **Provenance snapshot:** zip the exact retrieved pages (PDF slices) alongside the Markdown.
* **Prompt library:** YAML files per section with role + structure + citation style.

---

## 1.0) Decision‑Led Loop (v1)

* **Claim–Evidence–Finding (CEF) tables:** for each section, extract claims → retrieve counter‑evidence → write finding with confidence + policy hooks.
* **Planning Balance Matrix:** score effects against key policies/criteria with weights (planner‑tunable).
* **Reasoned departures:** detect when compliance is weak but benefits may justify; flag explicit rationale needed.
* **Uncertainty flags:** highlight thin evidence or conflicting sources.
* **Run registry:** each section records model versions, prompts, seeds, doc IDs, and hashes.

---

## Advanced Feature Catalogue

*(Keep for later development — don’t lose track of these ideas)*

**Retrieval / Chunking**

* LLM-guided dynamic chunking to respect semantic units (policies, chapters, DAS sections).
* Hybrid index: text (bge) + layout-aware signals; ColPali for page-level search with PDF thumbnails.
* Adversarial retrieval: deliberately fetch *contradictory* chunks (not just top similarity) to stress-test applicant claims.
* Source separation + mixing weights: prevent application docs from swamping policy context.
* Query planning: decompose a section into subqueries (policy tests, thresholds, standards).

**Knowledge Graph & Ontology**

* On-the-fly KG hints: extract nodes (Policy, Para, Topic, Site, Metric, Figure) + edges (supports/contradicts/refers-to/georefs).
* Material Considerations ontology: tag chunks with MC categories; drive section scaffolds from tags.
* Cross-references: auto-link policy paras to application statements; track citation frequency.

**Geospatial & Visual**

* Geo-ref extraction: NER for streets, postcodes, coordinates → map overlay hooks.
* Multimodal parsing: use `gemma3:27b` for images/figures/CGIs; summarize constraints, heights, massing.
* Layout-aware OCR: high-res OCR fallback for scanned PDFs; table extraction to structured CSV.
* Plan diagram vectorisation: detect footprints, heights, setbacks for later VLM checks.

**Scrutiny & Counterclaim Mining**

* Claim harvester over application statements (benefits, impacts, mitigations).
* Counter-evidence miner over policy text, inspector decisions (PINS), and technical standards.
* Consistency checks: detect internal contradictions across volumes/appendices.

**Precedent & External Evidence**

* PINS appeal decisions index: vector DB with tags (topic, outcome, policy cited, material weight).
* Benchmarking: retrieve comparable schemes for scale/density/parking precedents.

**Authoring & Orchestration**

* Section recipe files (YAML): inputs, subqueries, evidence mix, required tables/figures.
* Draft → critique loop: second-pass “Officer Critic” prompt to challenge weak findings.
* Style harmoniser: ensure tone, structure, and signposting match officer reports.

**Quality & Eval (planning‑specific)**

* Policy linkage check: each paragraph must cite at least one policy para/code, not only application docs.
* Balance presence check: every section ends with a mini planning-balance statement.
* Claim/evidence pairing: all applicant claims must be matched to supporting or counter policy evidence.
* Consistency check: ensure no section contradicts itself across retrieved sources.
* Citation sufficiency: flag any section where fewer than 3 distinct source IDs are cited.

**Ops & Repro**

* Caching: embedding + retrieval caches keyed by doc hash.
* Run snapshots: tarball of prompts, retrieved text, outputs, and a `RUN.yaml` manifest.
* Model registry: record exact model names (e.g., `gpt-oss:20b@sha`, `bge-large@rev`).

---

## Config you’ll tweak often

```yaml
retrieval:
  k_app: 6
  k_policy: 6
  k_adversarial: 4
  max_candidates: 60
  reranker: bge-reranker-large
  mix_weights: {app: 0.5, policy: 0.5}
sections:
  - transport
  - design_townscape
  - heritage
  - daylight_sunlight
  - ecology_bng
output:
  cite_style: inline_ids  # e.g., [APP:das_vol1_p123], [POL:lp_policy_sp7_para34]
  save_zip_of_pages: true
```

---

## Prompts (sketch)

* **Section Writer:** “Write the **{section}** assessment. Use retrieved evidence only. Cite source IDs inline. Separate **Claims**, **Evidence**, **Finding**, **Risks/Uncertainties**.”
* **Claim Harvester:** “Extract discrete claims (benefits/mitigations) with quoted spans + page IDs.”
* **Counter‑Evidence Miner:** “For each claim, retrieve opposing or constraining policy/test.”
* **Officer Critic:** “Challenge weak reasoning; demand specific policy para references.”

---

## What ‘good’ looks like

* Each paragraph has at least one verifiable citation ID.
* Applicant claims are paired with counter‑evidence or justification.
* Policy links are explicit (policy code + paragraph).
* A short planning-balance conclusion exists per section (not only in the final summary).

---

## 90‑Minute Sprint Plan (use when overwhelmed)

1. Implement **Minimal Flow** CLI.
2. Add source separation (policy vs application) and `k_*` knobs.
3. Write 3 prompt YAMLs (transport, heritage, design).
4. Generate 3 sections; save Markdown + RUN manifest + page zip.
5. Do a 10‑minute critique pass; log observations under **Findings & Gaps** below.

---

## Findings & Gaps

* Placeholder for issues noticed during critique passes.
* Example: “Transport section over‑relies on application docs; need stronger policy retrieval.”

---

## Parking Lot (Future Enhancements)

* VLM design-code checks (“in character” reasoning)
* Multi-pass retrieval curriculum (broad → specific)
* Precedent scoring vs current scheme
