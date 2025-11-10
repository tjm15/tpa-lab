# TODO

## High Priority (Officer-Grade Architecture)

### Critical: Material Considerations Engine

- [ ] **Implement Material Considerations (MC) as First-Class Objects**
  - [ ] Define MC schema in new `tpa/material_considerations.py`:
    ```python
    @dataclass
    class MaterialConsideration:
        topic: str  # 'heritage', 'transport', 'design', etc.
        policies: List[Policy]  # NPPF + Local Plan + SPD hierarchy
        statutory_consultees: List[str]
        applicant_evidence: List[ChunkRef]
        technical_evidence: List[ChunkRef]
        consultee_responses: List[ChunkRef]
        objections: List[ChunkRef]
        findings: List[Finding]  # claim → evidence → conclusion
        weight: str  # 'significant', 'moderate', 'limited', 'neutral'
        recommendation: str  # 'acceptable', 'unacceptable', 'requires_mitigation'
    ```
  - [ ] Auto-detect active MCs from document corpus
  - [ ] Build MC-specific retrieval plans per topic
  - [ ] Track MC dependencies and conflicts

- [ ] **MC Assessment Agent (per topic)**
  - [ ] Phase 1: Extract policy hierarchy (NPPF → Local Plan → SPD)
  - [ ] Phase 2: Extract applicant's case from DAS/statements
  - [ ] Phase 3: Stratified retrieval by source type:
    - applicant_case: {k: 5, weight: 0.3}
    - policy: {k: 8, weight: 0.3}
    - technical_report: {k: 5, weight: 0.2}
    - consultee: {k: 3, weight: 0.15}
    - objection/support: {k: 5, weight: 0.05}
  - [ ] Phase 4: Evaluate compliance with policy tests
  - [ ] Phase 5: Assess harm/benefit with weighting
  - [ ] Phase 6: Generate structured assessment + prose section

- [ ] **Recursive Section Agent with Multi-Stage Critique**
  - [ ] **Stage 1: Policy Framework Extraction**
    - Prompt: "List all applicable policies with hierarchy and requirements"
    - Output: Structured policy tree with thresholds/tests
    - Store in reasoning.json
  - [ ] **Stage 2: Claim-Evidence-Finding (CEF) Table**
    - Prompt: "Extract applicant claims; for each, identify supporting and contradicting evidence"
    - Include VLM analysis for visual claims (design quality, townscape impact)
    - Flag under-evidenced or contested claims
  - [ ] **Stage 3: Draft Generation**
    - Prompt: "Write officer assessment using CEF table + policy framework"
    - Style guide: "Cautious, evidenced, uses 'would' not 'will', acknowledges uncertainty"
    - Output: Section draft with inline citations [APP:x] [POL:y] [VIS:z]
  - [ ] **Stage 4: Critic Agent**
    - Check: Every claim has policy/evidence citation
    - Check: No contradictions with other sections
    - Check: All mandatory policies addressed
    - Check: Tone matches officer style (cautious, evidenced, balanced)
    - Check: Numbers/facts match source documents
    - Check: Consultee responses properly referenced
    - Output: List of issues with paragraph references
  - [ ] **Stage 5: Repair Agent**
    - Regenerate only flagged paragraphs
    - Maintain consistency with rest of section
    - Iterate until critic passes or max rounds (3-5)
  - [ ] Log all critique points and repairs in reasoning.json

- [ ] **Planning Balance Accumulator**
  - [ ] Each MC assessment outputs: {harms: [], benefits: [], neutral: [], weight: str}
  - [ ] Central agent tracks cumulative balance across all sections
  - [ ] Apply NPPF balancing tests (para 11, 202, 203) as applicable
  - [ ] Identify tipping points (e.g., heritage harm vs housing benefit)
  - [ ] Flag where "very special circumstances" or s38(6) departure needed
  - [ ] Generate final recommendation with explicit reasoning chain

- [ ] **MC Conflict Resolver**
  - [ ] Detect when MCs conflict (heritage harm vs housing need)
  - [ ] Apply policy hierarchy and statutory duties
  - [ ] Generate explicit trade-off reasoning
  - [ ] Flag where member discretion/judgment required
  - [ ] Cross-check consistency across sections

### Model Migration & Core Fixes

- [ ] **Switch to qwen3-VL:30b** for vision/multimodal processing (verified working on Ollama)
  - Replace `gemma3:27b` references in code
  - Update multimodal.py to use qwen3-VL:30b
  - **Enable VLM for subjective visual assessments**:
    - Design quality judgments from elevations/CGIs
    - Townscape impact from streetscape views
    - Heritage harm from context photos/maps
    - Amenity space quality from site plans/aerials
  - Generate vision evidence with clear reasoning: "The CGI shows..."
  - Test vision chunk integration end-to-end
  - Ensure `[VIS:...]` citations appear reliably in final output

- [ ] **Switch to qwen3-embedding:8b** for embeddings (via Ollama)
  - Replace `BAAI/bge-large-en-v1.5` / sentence-transformers
  - Update embed.py to call Ollama embedding API
  - Ensure no silent fallback to hash encoder

- [ ] **Debug embedding fallback** - ensure model loads or fail fast (no silent degradation)
  - Remove hash-based fallback or make it explicit error
  - Add startup validation that embedding model is accessible

- [ ] **Fix vision chunk integration** - VIS tags aren't reliable
  - Verify `[VIS:...]` citations appear in final output
  - Test with/without `TPA_DISABLE_VISION=1`
  - Ensure vision evidence is properly weighted in retrieval
  - **Add vision chunk types**:
    - Maps (site context, constraints, designations)
    - Diagrams (access arrangements, refuse strategy)
    - Elevations/sections (design assessment)
    - CGIs/photomontages (visual impact)
    - Site photos (existing character, amenity)

- [ ] **Test source mixing enforcement** - verify 70% cap works
  - Add unit test for mix_weights enforcement
  - Check that policy vs app balance is maintained
  - Log warnings when mixing rules are violated

- [ ] **Enable adversarial retrieval by default** (`k_adversarial: 2-4`)
  - Update CONFIG.sample.yml to set k_adversarial > 0
  - Test that contradictory evidence is actually retrieved
  - Document adversarial retrieval behavior in README

- [ ] **Add claim extraction for verification check #3**
  - Implement claim harvester in verify.py
  - Detect "the applicant states", quoted spans, benefit claims
  - Pair each claim with adjacent evidence citations

## Medium Priority (Spec Compliance & Evidence Architecture)

### Officer-Grade Report Structure

- [ ] **Adaptive Section Sizing Based on Contention**
  - [ ] Detect contention level from:
    - Number of objections mentioning topic
    - Conflict between applicant claims and policy
    - Consultee holding objections
    - Departure from policy required
  - [ ] Scale section length: 
    - Low contention: 200-400 words
    - Medium: 500-1200 words
    - High: 1500-3000 words with sub-headings
  - [ ] Add sub-section hierarchy for complex topics:
    ```
    ## Heritage
    ### Policy Framework
    ### Significance of Heritage Assets
    ### Impact Assessment
    ### Public Benefits
    ### Planning Balance
    ```

- [ ] **Officer Report Structure Templates**
  - [ ] Create section templates matching real officer reports:
    - Opening: "The main considerations are..."
    - Policy Framework: hierarchical policy quotes
    - Assessment: claim → evidence → finding pattern
    - Consultation Responses: "The Conservation Officer states..."
    - Objections: "Concerns have been raised that..."
    - Conclusion: "On balance..." with explicit weighting
  - [ ] Templates vary by section type (statutory consultee-led vs design-led vs quantitative)

- [ ] **Style Calibration Against Real Officer Reports**
  - [ ] Corpus of Real Officer Reports:
    - Scrape/collect 50-100 officer reports from planning portals
    - Extract by section (transport, heritage, design, etc.)
    - Annotate for: length, citation density, tone markers, structure
  - [ ] Style Metrics:
    - Hedging language frequency ("would", "likely", "on balance")
    - Policy citation rate (per paragraph)
    - Consultee quote integration
    - Quantitative evidence usage (numbers, measurements)
    - Paragraph length distribution
  - [ ] Few-Shot Prompting:
    - Include 2-3 real officer report excerpts in system prompt
    - Mark up with style features: "Note the use of 'would' for future impacts"
    - A/B test generated sections against real reports for style match
  - [ ] Tone Validator:
    - Check for prohibited patterns: 
      - Advocacy ("this excellent scheme")
      - Certainty where uncertain ("will definitely improve")
      - Missing caveats ("subject to conditions", "on balance")
    - Enforce officer voice: third person, passive constructions, evidenced claims

### Source-Type Aware Retrieval

- [ ] **Stratified Retrieval by Source Type**
  - [ ] Tag chunks by source type during indexing:
    ```python
    SourceType = Enum('applicant_case', 'policy', 'technical_report', 
                      'consultee', 'objection', 'support', 'precedent')
    ```
  - [ ] Retrieval recipe per section:
    ```yaml
    heritage:
      applicant_case: {k: 5, weight: 0.3}
      policy: {k: 8, weight: 0.3}  # NPPF + Local Plan
      consultee: {k: 3, weight: 0.2, required: ['conservation_officer']}
      objection: {k: 5, weight: 0.1, filter: 'heritage'}
      precedent: {k: 3, weight: 0.1, filter: 'listed_building'}
    ```
  - [ ] Enforce mandatory sources (e.g., Conservation Officer for heritage)
  - [ ] Flag missing sources as "cannot determine" in output

- [ ] **Consultee Response Parser**
  - [ ] Extract consultee type (statutory vs non-statutory)
  - [ ] Parse recommendation (no objection / holding objection / objection)
  - [ ] Extract conditions requested
  - [ ] Link to specific policy/material consideration
  - [ ] Auto-generate "Consultation" sub-sections

- [ ] **Objection/Support Summarizer**
  - [ ] Cluster objections by topic
  - [ ] Identify recurring concerns (e.g., "overlooking", "out of character")
  - [ ] Cross-reference with policy and technical evidence
  - [ ] Generate "Material Considerations Raised" paragraph

### Existing Spec Compliance Items

- [ ] **Implement query decomposition** per prompt schema
  - Break section queries into: claims → policy tests → findings
  - Add query_plan execution to author.py
  - Track sub-query results in reasoning.json

- [ ] **Add cross-reference tracking** (policy ↔ app statements)
  - Build citation graph during retrieval
  - Track which policy paras link to which app statements
  - Include in RUN.yaml manifest

- [ ] **Complete OCR fallback + table extraction**
  - Verify pytesseract integration works
  - Add table detection and CSV export
  - Test on scanned PDFs

- [ ] **Add reranker by default** (currently optional but SPECS implies expected)
  - Set `use_reranker: true` in CONFIG.sample.yml
  - Test bge-reranker-large performance
  - OR: switch to Ollama-based reranking if available


## Comprehensive Testing & Metrics

- [ ] **Unit tests for all core modules**
  - [ ] test_embed.py - embedding generation, index building, Ollama integration
  - [ ] test_retrieve.py - source mixing, adversarial retrieval, reranking, stratified retrieval
  - [ ] test_multimodal.py - vision chunk processing, VLM integration (qwen3-VL:30b)
  - [ ] test_author.py - CEF table generation, citation formatting, multi-stage drafting
  - [ ] test_verify.py - expand with all 5 checks + edge cases + claim extraction
  - [ ] test_package.py - manifest generation, evidence zipping
  - [ ] test_material_considerations.py - MC detection, assessment agent, conflict resolution

- [ ] **Integration tests**
  - [ ] End-to-end with real Ollama models (qwen3-VL:30b, qwen3-embedding:8b)
  - [ ] Test all sections: transport, heritage, design_townscape, daylight_sunlight, ecology_bng
  - [ ] Test recursive critique-repair loops
  - [ ] Verify output quality on multi-document corpus
  - [ ] Test MC assessment pipeline end-to-end

- [ ] **Metrics & Evaluation**
  - [ ] **Retrieval metrics**: precision@k, recall@k, source diversity, source-type distribution
  - [ ] **Citation metrics**: citations per paragraph, policy/app ratio, unique source count, VIS citation count
  - [ ] **Verification pass rate**: % of runs passing all 5 planner checks
  - [ ] **Quality metrics**: 
    - Claim-evidence pairing rate
    - Mini-balance presence
    - Consultee response coverage
    - Objection addressing rate
    - Policy hierarchy completeness
  - [ ] **Officer-grade metrics**:
    - Hedging language frequency ("would", "likely", "on balance")
    - Tone validator pass rate
    - Section length vs contention correlation
    - Critique iteration count
  - [ ] **Performance metrics**: indexing time, retrieval latency, LLM token usage, critique rounds
  - [ ] Add metrics output to RUN.yaml
  - [ ] Create metrics dashboard/report script

- [ ] **Regression testing**
  - [ ] Snapshot known-good outputs for comparison
  - [ ] Auto-detect degradation in citation quality
  - [ ] Test suite that runs on every commit
  - [ ] Compare against real officer report benchmarks

- [ ] **Stress testing**
  - [ ] Large corpus (100+ PDFs)
  - [ ] Long documents (500+ pages)
  - [ ] Scanned/poor quality PDFs
  - [ ] Documents with heavy graphics/tables
  - [ ] Highly contentious cases (50+ objections)
  - [ ] Complex policy hierarchies (NPPF + LP + multiple SPDs)

## Low Priority (Nice-to-Haves)

- [ ] **PINS appeal decisions index**
  - Scrape or ingest appeal decision corpus
  - Tag by topic, outcome, policy cited
  - Integrate into retrieval as precedent evidence

- [ ] **Benchmarking retrieval** (comparable schemes)
  - Find similar developments in database
  - Compare density, height, parking ratios
  - Surface as context in planning balance

- [ ] **GIS/geo-ref extraction**
  - NER for streets, postcodes, coordinates
  - Generate map overlay data
  - Link to constraint layers

- [ ] **Multi-pass retrieval curriculum**
  - First pass: broad topic retrieval
  - Second pass: specific policy/threshold retrieval
  - Third pass: adversarial/counter-evidence

## Parking Lot (Future Enhancements)

- [ ] **Precedent scoring vs current scheme**
  - Weight PINS decisions by similarity to current case
  - Factor into planning balance scoring

- [ ] **Full KG construction** (from specification.md)
  - Extract nodes: Policy, Para, Topic, Site, Metric, Figure
  - Build edges: supports/contradicts/refers-to/georefs
  - Use for enhanced retrieval and reasoning
  - Atom-based generation (distilled claims)
  - Multi-index fusion

---

## Architecture Evolution Map

```
┌─────────────────────────────────────────────────────┐
│ 1. INGEST & CHUNK (with source-type tagging)       │
│    → applicant_case / policy / consultee / etc.    │
│    → Vision chunks: maps, diagrams, elevations     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. MC DETECTION & PLANNING                          │
│    → Identify active material considerations        │
│    → Build MC-specific retrieval plans              │
│    → Detect contention level per MC                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. PER-MC ASSESSMENT LOOP (recursive)               │
│    For each MC:                                     │
│      a. Stratified retrieval (by source type)       │
│         - VLM analysis for visual evidence          │
│      b. Policy framework extraction                 │
│      c. Claim-Evidence-Finding table                │
│      d. Draft section (adaptive length)             │
│      e. Critique (policy/evidence/tone/consistency) │
│      f. Repair (iterate 2-5x until pass)            │
│      g. Output: {prose, findings, weight}           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4. CROSS-SECTION CONSISTENCY CHECK                  │
│    → Detect contradictions between MCs              │
│    → Resolve via MC Conflict Resolver               │
│    → Ensure consistent tone/style across sections   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 5. PLANNING BALANCE SYNTHESIS                       │
│    → Aggregate harms/benefits with weights          │
│    → Apply NPPF tests (para 11, tilted balance)     │
│    → Apply heritage tests (para 202, 203)           │
│    → Generate recommendation + reasoning chain      │
│    → Identify conditions/obligations needed         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 6. FINAL ASSEMBLY & PACKAGING                       │
│    → Stitch sections with consistent tone           │
│    → Add conditions/obligations/informatives        │
│    → Package with full evidence bundle + VIS chunks │
│    → Generate RUN.yaml with full audit trail        │
└─────────────────────────────────────────────────────┘
```

---

## Notes

**Current Architecture Blockers:**
- Silent embedding fallback to hash encoder causing poor retrieval
- Vision integration incomplete/unreliable (VIS tags not appearing)
- Default config uses DummyEcho LLM instead of real model
- Insufficient test coverage to catch regressions
- **No MC-based reasoning** - current system is single-pass summarizer, not decision-led
- **No recursive critique loops** - drafts aren't challenged or refined
- **No source-type stratification** - treats all evidence equally
- **No officer-grade style calibration** - tone/length don't match real reports

**Officer-Grade Evolution Path:**
1. ✅ Switch to qwen3-VL:30b for VLM (subjective visual assessments)
2. ✅ Switch to qwen3-embedding:8b for embeddings
3. 🔲 Implement Material Considerations as first-class objects
4. 🔲 Build stratified retrieval (applicant/policy/consultee/objection)
5. 🔲 Implement recursive critique-repair loops (2-5 iterations)
6. 🔲 Add adaptive section sizing based on contention
7. 🔲 Calibrate style against real officer reports (corpus of 50-100)
8. 🔲 Build Planning Balance Accumulator (cross-section synthesis)
9. 🔲 Add MC Conflict Resolver for contradictory findings

**Model Migration Path:**
1. Update embed.py to call Ollama embedding API for qwen3-embedding:8b
2. Update multimodal.py to use qwen3-VL:30b instead of gemma3:27b
3. Enable VLM for: design quality, townscape impact, heritage harm, amenity assessment
4. Update CONFIG.sample.yml with new model names
5. Add model availability check on startup
6. Update README with new model requirements and VLM capabilities
