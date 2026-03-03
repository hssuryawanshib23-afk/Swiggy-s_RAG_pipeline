# Swiggy RAG Pipeline

## Vision

Build a production-quality RAG (Retrieval-Augmented Generation) system that can accurately answer questions from Swiggy's Annual Report FY 2023-24 — a complex PDF containing mixed content types.

## Problem

The PDF contains selectable text, scanned image pages, tables, charts, flowcharts, infographic cards, handwritten page numbers, and landscape-oriented pages. Most naive PDF parsers miss 40%+ of the content, leading to a RAG system that can't answer financial questions from the critical pages (32+).

## Core Value

**Accurate grounded answers from the entire PDF** — including scanned financial statements that most candidates miss.

## Target Users

- Evaluators reviewing this ML intern assignment
- Anyone needing instant answers from complex financial PDFs

## Constraints

- Free-tier tools only (no paid APIs beyond minimal usage)
- Must demonstrate OCR handling of scanned pages
- Must show source attribution (page numbers)
- Must include anti-hallucination guardrails

## PDF Content Breakdown

| Content Type | Pages | Extraction Method |
|---|---|---|
| Selectable text | 1-31 (mostly) | pdfplumber |
| Scanned image pages (not machine-readable) | 32 → end | pytesseract OCR |
| Tables (text + numbers) | Throughout | pdfplumber table extraction |
| Financial totals/subtotals | Throughout | pdfplumber + custom parsing |
| Flowcharts (selectable) | Various | pdfplumber text |
| Infographic cards / roadmap | pg 7, 8 | OCR fallback |
| Bar charts with numbers | Various | OCR |
| Images with embedded text (org charts) | pg 4 | OCR |
| Handwritten page numbers | Throughout | Ignore / OCR |
| Landscape pages | Mixed | Rotation-aware extraction |

## Tech Stack

| Component | Tool | Rationale |
|---|---|---|
| Text extraction | pdfplumber | Best for selectable text + tables |
| Image detection | pymupdf (fitz) | Detects image-heavy pages for OCR routing |
| OCR | pytesseract | Handles scanned pages (pg 32+) |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, high quality |
| Vector DB | FAISS | Local, fast, no API cost |
| LLM | Groq API (Llama 3.3 70B) | Free tier, fast inference |
| Framework | LangChain | Standard RAG orchestration |
| UI | Streamlit | Fast to build, clean demo |

## Key Decisions

| Decision | Rationale | Outcome |
|---|---|---|
| Hybrid extraction (pdfplumber + OCR) | PDF has both selectable and scanned pages | Smart auto-routing |
| FAISS over cloud vector DBs | Free, local, easy to demo | No vendor lock-in |
| Groq free tier over OpenAI | Free, runs Llama 3.3 70B, fast | Zero cost |
| 512-token chunks with 50-token overlap | Good balance for financial docs | Captures context without noise |
| Strict grounding prompt | Anti-hallucination is critical for financial data | Prevents fabricated numbers |

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Hybrid PDF extraction (selectable text + OCR for scanned pages)
- [ ] Table extraction with structure preservation
- [ ] Smart page routing (auto-detect scanned vs selectable)
- [ ] Semantic chunking with section metadata
- [ ] Vector embedding and FAISS indexing
- [ ] RAG query pipeline with source attribution
- [ ] Anti-hallucination grounding prompt
- [ ] Streamlit UI with question input + answer + source display

### Out of Scope

- Multi-PDF support — single PDF focus for assignment
- Authentication / user management — demo only
- Fine-tuning models — using pre-trained embeddings
- Cloud deployment — local demo

---
*Last updated: 2026-03-03 after initialization*
