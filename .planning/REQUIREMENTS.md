# Requirements — Swiggy RAG Pipeline

## Core Value

Accurate, grounded answers from a complex mixed-content PDF (Swiggy Annual Report FY 2023-24).

---

## Phase 1: PDF Ingestion & Extraction

### Must Have
- [ ] Hybrid extraction: pdfplumber for selectable text, pytesseract for scanned pages
- [ ] Auto-detection: if `len(text.strip()) < 50` → route to OCR
- [ ] Table extraction via `pdfplumber.extract_tables()` with `vertical_strategy: "text"` for borderless tables
- [ ] Landscape page handling (rotation-aware extraction)
- [ ] Clean OCR output (remove garbled chars, fix spacing)

### Should Have
- [ ] Section header detection for metadata tagging
- [ ] Page-level content type classification (text / table / image / mixed)

### Won't Have (this milestone)
- Multi-PDF support
- Real-time PDF upload (pre-processed only)

---

## Phase 2: Chunking & Embedding

### Must Have
- [ ] Semantic chunking: 512 tokens, 50-token overlap
- [ ] Metadata per chunk: `{ page, section, type: "text"|"table"|"ocr_text" }`
- [ ] RecursiveCharacterTextSplitter with `["\n\n", "\n", ".", " "]`
- [ ] BAAI/bge-small-en-v1.5 embeddings
- [ ] FAISS vector store creation and persistence

### Should Have
- [ ] Separate table chunks (preserve table structure as markdown)
- [ ] Deduplication of near-identical chunks

---

## Phase 3: RAG Query Pipeline

### Must Have
- [ ] Retrieval: top-k relevant chunks from FAISS
- [ ] Strict grounding prompt (answer ONLY from context)
- [ ] Source attribution: show page number + section for each answer
- [ ] Groq API integration (Llama 3.3 70B, free tier)
- [ ] Graceful "not found" responses when context lacks answer

### Should Have
- [ ] Re-ranking of retrieved chunks before LLM call
- [ ] Multi-hop retrieval for complex questions

---

## Phase 4: Streamlit UI

### Must Have
- [ ] Text input for user questions
- [ ] Answer display box
- [ ] Source chunks expander (page, section, chunk text)

### Should Have
- [ ] Chat history
- [ ] Loading spinner during retrieval
- [ ] Example questions sidebar

---
*Last updated: 2026-03-03 after initialization*
