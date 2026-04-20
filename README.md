\# Warhammer 40K RAG System



A fully local Retrieval-Augmented Generation (RAG) system that answers Warhammer 40,000 rules questions using a local vector database of 85+ faction PDFs and Claude (Anthropic API) for answer generation.



Built as a personal learning project to explore RAG architecture, PDF extraction pipelines, vector embeddings, and local LLM infrastructure.



\---



\## What It Does



\- Ingests 85 Warhammer 40K rulebooks and faction PDFs into a local ChromaDB vector store

\- Uses a smart sub-page segmentation pipeline to classify and extract content by region type (stat blocks, rules text, narrative, artwork, etc.)

\- Serves a locally hosted web app where you can ask natural language questions about rules, units, weapons, stratagems, and faction abilities

\- Uses Claude (claude-sonnet) via the Anthropic API to generate accurate, grounded answers with source citations

\- Streams responses token-by-token in real time



\---



\## Architecture



```

User Query

&#x20;   │

&#x20;   ▼

FastAPI Backend

&#x20;   │

&#x20;   ├── Claude API (filter extraction)

&#x20;   │       └── Extracts faction + category filters from query

&#x20;   │

&#x20;   ├── ChromaDB (semantic search)

&#x20;   │       └── nomic-embed-text via Ollama

&#x20;   │           Returns top N relevant chunks

&#x20;   │

&#x20;   └── Claude API (answer generation)

&#x20;           └── Streams grounded answer with source citations

```



\---



\## Tech Stack



| Layer | Technology |

|---|---|

| \*\*PDF Extraction\*\* | PyMuPDF, pdfplumber, Tesseract OCR, Poppler |

| \*\*Chunking\*\* | Custom sub-page region segmenter (3-pass geometric + content detection) |

| \*\*Embeddings\*\* | `nomic-embed-text` via Ollama (fully local) |

| \*\*Vector DB\*\* | ChromaDB (persistent, in-process) |

| \*\*LLM\*\* | Claude `claude-sonnet-4` via Anthropic API |

| \*\*Backend\*\* | FastAPI + uvicorn |

| \*\*Frontend\*\* | Vanilla HTML/CSS/JS (dark WH40K theme) |

| \*\*Runtime\*\* | Python 3.11, Windows 11, NVIDIA RTX 3070 |



\---



\## Project Structure



```

wh40k-app/

├── ingest.py                    # PDF ingestion pipeline → ChromaDB

├── main.py                      # FastAPI backend + RAG logic

├── test\_retrieval.py            # Retrieval quality diagnostic tool

├── .gitignore

├── static/

│   └── index.html               # Chat UI

└── pdf\_agent/

&#x20;   ├── pdf\_agent.py             # PDF assessment + extraction strategies

&#x20;   ├── pdf\_region\_segmenter.py  # Sub-page region detection (3-pass)

&#x20;   └── assess\_to\_csv.py         # PDF assessment diagnostic tool

```



\---



\## Prerequisites



\### System Requirements

\- Windows 10/11

\- Python 3.11

\- NVIDIA GPU recommended (for Ollama embedding speed)



\### Required System Installs

| Tool | Purpose | Download |

|---|---|---|

| \*\*Ollama\*\* | Local embedding model server | \[ollama.com](https://ollama.com/download) |

| \*\*Tesseract OCR\*\* | OCR for image-heavy PDFs | \[UB Mannheim](https://digi.bib.uni-mannheim.de/tesseract/) |

| \*\*Poppler\*\* | PDF image extraction tools | \[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) |



\### Ollama Models

After installing Ollama, pull the embedding model:

```bash

ollama pull nomic-embed-text

```



\---



\## Setup



\*\*1. Clone the repo\*\*

```bash

git clone https://github.com/MrP32/wh40k-rag.git

cd wh40k-rag

```



\*\*2. Create and activate a virtual environment\*\*

```powershell

\& "C:\\Program Files\\Python311\\python.exe" -m venv .venv

.venv\\Scripts\\Activate.ps1

```



\*\*3. Install dependencies\*\*

```powershell

pip install fastapi uvicorn anthropic chromadb python-dotenv pymupdf pdfplumber pypdf pdf2image pytesseract pillow reportlab openpyxl requests ollama

```



\*\*4. Create your `.env` file\*\*

```

ANTHROPIC\_API\_KEY=your\_api\_key\_here

```

Get your API key at \[console.anthropic.com](https://console.anthropic.com)



\*\*5. Add your PDF source files\*\*



Place your Warhammer 40K PDFs in a folder of your choice and update the `PDF\_FOLDER` path in `ingest.py`:

```python

PDF\_FOLDER = r"C:\\path\\to\\your\\pdfs"

```



\---



\## Running the Pipeline



\### Step 1 — Start Ollama

Make sure Ollama is running (check system tray or run `ollama serve`)



\### Step 2 — Ingest PDFs into ChromaDB

```powershell

python ingest.py

```

This will take 30–90 minutes depending on the number of PDFs and your hardware. It runs a 3-pass sub-page region segmentation pipeline across every PDF.



\### Step 3 — Start the web app

```powershell

uvicorn main:app --reload

```



\### Step 4 — Open the app

Navigate to `http://localhost:8000` in your browser.



\---



\## How the Ingestion Pipeline Works



Each PDF goes through a multi-stage pipeline:



\*\*1. Assessment\*\* — Classifies each PDF by type:

\- `text` → clean text layer

\- `table-heavy` → table-first extraction

\- `mixed` → text pages + OCR image pages

\- `graphic-stats` → fully visual datasheets (WH40K stat blocks)

\- `scanned` → full OCR



\*\*2. Sub-page Region Detection (3 passes)\*\*

\- \*\*Pass 1\*\* — Geometric boundary detection (image bboxes, rect dividers, whitespace gaps)

\- \*\*Pass 2\*\* — Content signal verification (crops each region, re-classifies by content type)

\- \*\*Pass 3\*\* — Per-region extraction routing (best method per section type)



\*\*3. Section Classification\*\* — Each chunk is classified as one of:

`unit\_datasheet` | `stratagem` | `objective` | `ability` | `enhancement` | `rules` | `narrative` | `general`



\*\*4. Metadata Enrichment\*\* — Every chunk carries:

\- `content\_category` — what type of document it came from

\- `subject` — which faction or rulebook

\- `section\_identifier` — the heading it belongs to (carried forward into all chunks in that section)

\- `extraction\_method` — how the text was extracted

\- `ocr\_confidence` — confidence score for OCR-extracted chunks



\---



\## How Retrieval Works



Each user query goes through two Claude API calls:



\*\*1. Filter extraction\*\* — A lightweight Claude call extracts structured metadata filters from the query:

```json

{"subject": "space marines", "category": "faction\_rules"}

```



\*\*2. Semantic search\*\* — ChromaDB searches the filtered subset using nomic-embed-text embeddings, returning the top N most relevant chunks.



\*\*3. Answer generation\*\* — The retrieved chunks are injected into a system prompt and Claude generates a grounded, cited answer streamed token-by-token.



\---



\## Key Design Decisions



\*\*Why local embeddings?\*\*

`nomic-embed-text` via Ollama runs entirely on local GPU/CPU — no data leaves the machine during indexing or retrieval.



\*\*Why sub-page region segmentation?\*\*

A single Warhammer datasheet page can contain unit stat blocks, weapon tables, ability rules, and lore text — all requiring different extraction methods. Page-level chunking treats them identically and loses structure.



\*\*Why section heading carry-forward?\*\*

When a page is split into regions, the section heading lands in one chunk and the rules text lands in the next. Without carry-forward, rules chunks have no identifying context and semantic search can't find them. Prepending `\[LIBRARIUS CONCLAVE]` to every downstream chunk from that section makes retrieval dramatically more reliable.



\*\*Why query rewriting instead of hardcoded filters?\*\*

Hardcoded faction/category dictionaries break for unexpected queries and require constant maintenance. Using Claude to dynamically extract filters handles any query including ones never anticipated.



\---



\## Known Limitations



\- OCR quality varies on heavily stylised pages (two-column layouts can produce garbled text)

\- Very large faction packs (Space Marines, 290+ chunks) can dilute semantic search results

\- Re-ingestion takes 30–90 minutes — incremental ingestion (skip unchanged PDFs) is a planned improvement



\---



\## Roadmap



\- \[ ] Incremental ingestion — checksum-based skip for unchanged PDFs

\- \[ ] Hybrid BM25 + semantic search for exact term matching

\- \[ ] FastAPI hardening — API key authentication, CORS

\- \[ ] Network security hardening

\- \[ ] Cloud deployment via Docker + Cloudflare Tunnel



\---



\## License



Personal learning project — not affiliated with Games Workshop or Warhammer 40,000.

