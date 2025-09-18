# ECDC Dynamic Report Pipeline — Decision‑Led Spec (v0.1)

**Stance (from Tim’s decisions):**

* Semantic/structural chunking (incl. images/diagrams as first‑class chunks).
* Enrich during chunking (front‑load plausibility; reproducibility is secondary).
* Knowledge‑graph‑assisted retrieval (KG expands/steers context).
* Context assembly is fully probabilistic (no hard rule filters unless obvious).
* Report sections adapt from a high‑level template.
* Recursive, small‑section generation (per‑section agent loop).
* Final balance = meta‑reasoning pass over all sections.

---

## 0) Glossary (concrete object model)

* **Doc**: The original source (e.g., DAS vol X, ES chapter Y, RBKC Local Plan policy box, NPPF para, appeal decision).
* **Chunk**: Small, semantically coherent unit derived from a Doc. Types: `text`, `table`, `figure`, `map`, `diagram`, `plan`, `policy`, `precedent`.
* **Atom**: A distilled claim/evidence line extracted from one or more Chunks with a pointer back to exact spans (used inside generations).
* **Issue**: A planning topic label (e.g., Housing mix, Design/Character, Heritage, Transport, BNG, Flood, Daylight/Sunlight, Amenity, Affordable, Energy, Access, Townscape, Tall buildings) used for section detection.
* **KG (knowledge graph)**: Nodes = {Doc, Chunk, Policy, Issue, Entity, Site/Geom, Precedent}. Edges = typed relations (e.g., `refers_to_policy`, `supports`, `contradicts`, `about_issue`, `located_in`, `has_figure`, `quantifies`, `requires`). Edges have `weight` (confidence) and optional `evidence` pointers.

---

## 1) Enrichment During Semantic Chunking (incl. images)

**Goal:** Produce retrieval‑ready Chunks with sufficient metadata + opportunistic KG links.

### 1.1 Parsers

* **PDF/Text parser**: detect headings, numbered clauses, policy boxes, tables, lists; retain section numbering and anchors.
* **Layout parser**: capture page layout (x/y/width/height) for context windows and figure‑caption association.
* **Image/Diagram extractor**: each image/diagram becomes a `figure` Chunk with: raster (thumb), OCR/alt‑text, caption, detected elements (e.g., massing, heights, annotations), bounding boxes of labels, detected legend.
* **Policy parser**: canonicalises policy IDs (e.g., RBKC `H3`, LBHF `HS7`, London Plan `H10`), extracts targets/requirements lists, and policy hierarchy (Local > London Plan > NPPF). Marks per‑criterion line items.
* **Precedent parser**: extracts appeal/site refs, outcome, material points; normalises to Issues.

### 1.2 Chunk schema (minimum viable)

```json
{
  "chunk_id": "c_...",
  "doc_ref": "DAS_Vol1_Ch3",
  "type": "text|policy|figure|table|map|diagram|precedent",
  "title": "H10 Housing Mix – Clause B",
  "section_path": ["3", "3.2", "(b)"],
  "text": "...",            
  "tokens": 512,
  "page": 47,
  "bbox": [x,y,w,h],
  "caption": "Figure 13: Massing strategy",
  "themes": ["housing_mix", "tall_buildings"],
  "policy_ids": ["LP_H10", "LP_D12"],
  "entities": ["Earl's Court Site", "RBKC"],
  "numbers": [{"label":"units_1bed","value":320,"unit":"dwellings"}],
  "geom_refs": ["site_boundary", "character_area_A"],
  "cross_refs": [{"kind":"policy","id":"LP_H10","anchor":"(B)(1)"}],
  "embeddings": {"text": "...", "image": "...", "layout": "..."}
}
```

### 1.3 KG construction (opportunistic)

* Create nodes for policies (unique per authority + version), chunks, issues, entities, docs, precedents.
* Add edges while parsing:

  * `chunk ->policy (refers_to_policy)` when ID pattern or semantic match.
  * `chunk ->issue (about_issue)` via multi‑label classifier.
  * `figure ->chunk (has_figure)` when caption/nearby text ties them.
  * `chunk ->precedent (cites)` when PINS IDs detected.
  * `policy ->policy (cross_refers)` from explicit cross‑refs.
* Keep `weight` and `source` (regex, VLM, NER, heuristic) for each edge. Store cheaply (JSONB) but index for neighborhood queries.

### 1.4 Image/diagram handling (first‑class)

* For each figure/map/diagram:

  * Run OCR; associate nearby caption; VLM captioning for alt‑text; detect structured items (e.g., storeys, heights, blocks, viewpoints labels).
  * Generate image embeddings; store detected symbols as `atoms` (e.g., `max_height_block_A = 32 storeys`).
  * Link figure→relevant issues (`tall_buildings`, `townscape`, `daylight_sunlight`).

**Outcome:** a retrieval‑rich corpus where each Chunk is small, typed, themed, and connected; images are queryable as evidence.

---

## 2) Adaptive Report Assembly

**High‑level:** Detect issues → for each active section, assemble probabilistic context → generate recursively with provenance → finalize with balance meta‑agent.

### 2.1 Template & Section Detection

* **Template scaffold** (order is suggestive, not prescriptive):

  1. Proposal summary & site context
  2. Policy framework summary (Local/London/NPPF)
  3. Issue sections (dynamic subset): Housing mix & affordability; Design/townscape & tall buildings; Heritage; Transport & access; Amenity; Environmental (BNG, flood, energy); DS/OVD; Social infrastructure; Phasing & delivery; Conditions & obligations (HoTs/viability pointers).
  4. Planning balance & recommendation.
* **Activation**: Run an issue detector over all Chunks (probabilistic). Select sections whose score > τ or whose policy nodes are densely connected to application chunks in the KG. Allow a small tail of low‑probability sections to avoid false negatives.

### 2.2 Context Assembly (probabilistic + KG‑steered)

* **Query recipe per section** (no hard filters):

  1. Build a **section query vector** from the section descriptor (e.g., "London Plan H10 housing mix for large regeneration sites in RBKC/LBHF"), enriched by:

     * policy nodes adjacent in KG (1–2 hop expansion, weighted),
     * site/geom nodes (constraint overlays if available),
     * previously generated atoms in earlier sections (for coherence).
  2. Retrieve top‑k from **multi‑index fusion** (text, image, layout). Use reciprocal‑rank fusion; diversify by doc and type (MMR).
  3. Expand via **KG neighborhood** of top results (add near policies/precedents). Re‑rank by edge weights and novelty.
  4. Distil to **Atoms**: extract 5–20 short evidence atoms with explicit citations (chunk\_id + char spans) + confidences.

### 2.3 Recursive Section Generation (agent loop)

For each active section:

* **Draft step**: LLM writes a tight narrative using Atoms only; every claim must cite atoms. Limit to \~500–900 words.
* **Critic step**: a verifier agent checks: unsupported claims, contradictions with earlier sections, missing mandatory policy mentions (detected by KG density), ambiguous numbers. It proposes edits and atom‑level additions.
* **Repair step**: regenerate only affected paragraphs; re‑verify.
* **Outputs**: `section_text`, `section_summary_bullets`, `compliance_signals` (per‑policy qualitative signals), `citations` (chunk\_id + spans), `open_questions` (gaps to flag).

### 2.4 Provenance Weaving

* Inline numbered citations `[c12]` mapping to chunk ids and page anchors; figures referenced like `Fig. 13` with thumbnail links.
* Store a **section bundle**: `{prompt, retrieved_chunks, atoms, drafts, critic_notes, final_text}` for auditability without heavy reproducibility overhead.

---

## 3) Balance Meta‑Agent (final pass)

* **Inputs**: all section summaries, compliance\_signals, key numbers (units, heights, harms/benefits), unresolved questions, and KG signals about policy hierarchy.
* **Process**:

  1. Consolidate duplicates/contradictions; force explicit trade‑offs (e.g., heritage harm vs. public benefits)
  2. Weight by **policy hierarchy** and **materiality** (learned heuristics from prior officer reports/appeals; adjustable sliders in UI later).
  3. Draft the planning balance narrative; propose **conditions/obligations**; state **recommendation**.
* **Output**: `balance_text`, `conditions_skeleton`, `recommendation`, `residual_risks`.

---

## 4) Retrieval Details (multi‑index fusion)

* Maintain parallel indices: `text_emb`, `image_emb`, `layout_emb` (optional), and a lightweight KG store.
* **Fusion**: score = RRF(text\_rank, image\_rank, kg\_boost), with novelty/diversity via MMR.
* **De‑dup**: HDBSCAN or cosine‑near duplicate pruning to avoid citing the same passage 5 times.
* **Negative sampling** at training time for any learned re‑ranker to reduce policy name spurious matches.

---

## 5) Failure Modes & Mitigations

* **Hallucination** → Atom‑only generation; critic agent enforces citations.
* **Section drift** (mixing issues) → keep a section‑specific vocabulary and policy whitelist from KG.
* **Missing key policy** → KG density check; critic step demands mention or explicit absence note.
* **Over‑citation of policy headlines** → re‑rank toward clause‑level chunks (policy criterion atoms).
* **Figure misinterpretation** → prefer figure caption + nearby text; mark VLM‑only inferences as lower‑confidence unless corroborated.

---

## 6) Evaluation & Guardrails (lightweight, planner‑friendly)

* **Claim check set**: small curated spreadsheet of must‑hit facts per section (e.g., total units, affordable %, max height). Verifier confirms presence & correct citation.
* **Contradiction scan** across sections using NLI on section summaries.
* **Planner overrides**: allow manual pin/unpin of key chunks; the agent must incorporate pinned atoms next round.

---

## 7) Implementation Plan (thin‑slices)

**MVP‑1: Corpus & KG**

* Implement semantic chunker + image extraction for DAS + a subset of Local Plan + London Plan policies.
* Attach metadata + build opportunistic KG edges.
* Build multi‑index (text/image) + simple KG neighborhood API.

**MVP‑2: Section loop**

* Hardcode 5 issues (Housing mix, Design/Townscape, Heritage, Transport, BNG).
* Implement retrieval recipe + atom distillation + draft→critic→repair loop.
* Inline citations + section bundle persistence.

**MVP‑3: Balance pass**

* Aggregate section signals; write balance narrative; output recommendation + conditions skeleton.

**MVP‑4: UX hooks**

* Pinned chunks; figure thumbnails; per‑section “evidence table”; export to DOCX/PDF.

---

## 8) Open Decisions (defer until needed)

* **Embeddings**: which text/image models; dimensionality; whether to train a light re‑ranker.
* **KG store**: JSONB in Postgres vs. a tiny graph db vs. in‑memory typed edge lists; start JSONB.
* **Atom extractor**: regex + few‑shot vs. structured prompting; start with few‑shot prompt + numeric regex.
* **Figure understanding**: VLM selection; when to trust numeric readings from diagrams.
* **Section activation threshold τ** and max k per retrieval stage.

---

## 9) Interfaces (sketch)

### 9.1 Retrieve context for a section

```http
GET /sections/{issue}/context?site=ECDC&k=40
→ { "query_vec": "...", "seed_policies": ["LP_H10"],
    "top_chunks": [...], "kg_neighbors": [...], "atoms": [...] }
```

### 9.2 Generate section

```http
POST /sections/{issue}/generate
{ "atoms": [...], "prior_sections": ["design", "housing"], "constraints": {} }
→ { "text": "...", "summary": ["..."], "signals": {...}, "citations": [...] }
```

### 9.3 Balance

```http
POST /balance
{ "sections": [{"issue":"heritage","summary":"...","signals":{...}}] }
→ { "balance_text": "...", "conditions": [...], "recommendation": "grant|refuse|defer" }
```

---

## 10) Example Walkthrough (Housing Mix — London Plan H10)

1. **Detect section** via issue detector.
2. **Context recipe** builds query from descriptor + KG hops from `LP_H10` and local policy nodes; retrieve 20 text chunks + 4 figures; diversify; expand neighbors (appeal cites about mix on regeneration sites).
3. **Atoms** distilled:

   * `Total units = 4125 (DAS v1 p.47) [c812]`
   * `Mix target for 2-3 bed units ≥ 30% (LP_H10(B)(1)) [p_H10_B1]`
   * `Proposed 1-bed = 32% (Table 4.3) [c119]`
   * `Proposed 3-bed = 14% (Table 4.3) [c119]`
4. **Draft** cites atoms; **Critic** flags missing local policy clause; **Repair** adds RBKC equivalent and notes deviation.
5. **Output**: concise narrative + bullet summary + signals `{LP_H10: partial_compliance, RBKC_H3: non_compliant}`.

---

## 11) Prompts (minimal, enforce atoms)

**Section draft (system rule excerpt):**

> You must only state claims that are directly supported by the provided ATOMS. Every sentence containing a fact includes a citation like \[c###] mapping to chunk spans. Prefer clause‑level policy citations over section headings. If evidence is missing, write a one‑line TODO under “Open questions” instead of guessing.

**Critic rule (excerpt):**

> Reject paragraphs with uncited facts. Ensure at least one local policy + one higher‑tier policy is cited if KG density suggests relevance. Flag contradictions with previously accepted section summaries.

---

### Done‑ness for v0.1

This spec encodes the decisions you made and leaves model/tool choices open. Next step is MVP‑1: implement the semantic chunker (incl. figures), attach metadata, and build the opportunistic KG with JSONB edges.
