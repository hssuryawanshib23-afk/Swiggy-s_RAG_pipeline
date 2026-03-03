# Roadmap — Swiggy RAG Pipeline

## Milestone 1: Working RAG System


### Phase 3: RAG Query Pipeline [COMPLETED]
**Goal:** End-to-end query → retrieval → LLM → grounded answer with sources.
**Deliverable:** Working RAG chain with Groq API.
**Key risk:** Hallucination on financial numbers — must enforce strict grounding.

### Phase 4: Streamlit UI [COMPLETED]
**Goal:** Clean demo UI with question input, answer display, and source attribution.
**Deliverable:** Deployable Streamlit app.
**Key risk:** None significant — straightforward UI.

---

## Phase Dependencies

```
Phase 1 (Extraction) → Phase 2 (Chunking) → Phase 3 (RAG) → Phase 4 (UI)
```

All phases are sequential — each depends on the previous.

---

## Current Phase: **COMPLETED** 🎉

---
*Last updated: 2026-03-03 after Phase 5 completion*
