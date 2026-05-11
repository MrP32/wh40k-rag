# feature/desktop-ui — Branch Notes

> *"Knowledge is power. Guard it well."*

## What this branch does

Adds a desktop-app feel on top of the existing FastAPI + ChromaDB backend,
without touching the retrieval pipeline. Two user-facing changes:

1. **`launch.bat`** — click-to-run launcher. Auto-starts Ollama, activates
   the venv, runs uvicorn bound to localhost only, opens the browser.
2. **Blood Ravens themed UI** — floating centered chat card, dark gothic
   palette, rotating mottos in the empty state, clickable PDF source chips
   that open the source document at the cited page.

## Files changed

| File              | Status   | Notes                                                                                                |
|-------------------|----------|------------------------------------------------------------------------------------------------------|
| `launch.bat`      | NEW      | Ollama health check + auto-start, venv activation, uvicorn launch, browser open.                     |
| `main.py`         | MODIFIED | Added `/pdf/{filename}` endpoint, `sources` event in chat stream, `PDF_FOLDER` + `ANTHROPIC_MODEL` env vars. |
| `static/index.html` | REPLACED | Full UI redesign — Blood Ravens theme, floating card, source chips, rotating mottos.                |
| `.env.example`    | NEW      | Template for all env vars, including `PDF_FOLDER` which is needed by both `main.py` and `ingest.py`. |

## What is intentionally NOT in this branch

These were discussed and parked to keep scope tight:

- **Mode toggle (40K vs Combat Patrol).** Triggers a metadata-contract change
  where the UI sets authoritative filters that override the LLM router. Worth
  doing — needs to be designed alongside the schema-cleanup work.
- **Faction picker / sidebar.** Same reason. Faction filtering already works
  via the LLM router; a UI picker would add a second control plane that needs
  precedence rules.
- **Live "dataset changed" awareness.** Requires a `/factions` endpoint that
  queries Chroma for distinct subject values and a UI that refetches it.
  Easy mechanically, but coupled to the schema-cleanup work since the
  "subject" vocabulary is currently inconsistent between `ingest.py` and the
  filter prompt.
- **Backend ingestion trigger from UI.** Synchronous 90-minute jobs from a
  web request is the wrong pattern. Needs a background-job design.
- **`ingest.py` env-var conversion.** `PDF_FOLDER` is now in `.env.example`
  and `main.py` reads it. `ingest.py` still has the hardcoded path. Lifting
  it is a 2-line change but belongs with the broader reproducibility pass
  (`requirements.txt`, etc.) rather than as a drive-by.

## Design notes

**Why localhost-only by default.** `launch.bat` runs uvicorn with
`--host 127.0.0.1` explicitly. The app holds an Anthropic API key and serves
PDFs from your local filesystem; it should not be accidentally exposed to
your LAN. If you ever want LAN access, change the host flag intentionally
and add the auth/CORS work from the roadmap.

**Path-traversal protection on `/pdf/{filename}`.** Only accepts a leaf
filename (no slashes, no `..`), resolves it against `PDF_FOLDER`, and uses
`Path.relative_to()` to confirm the resolved path is still inside the
folder. Without this, `GET /pdf/..%2F..%2FWindows%2Fsystem32%2Fdrivers...`
could read arbitrary files.

**Why `Content-Disposition: inline`.** Without this header some browsers
download PDFs instead of rendering them in-tab. Inline tells the browser to
use its built-in viewer, which respects `#page=N` fragments to jump to a
specific page.

**Why the sources event comes after the answer stream.** The frontend
renders tokens as they arrive. If sources came first, the UI would need to
hold them until generation finished anyway. Emitting them at the end keeps
the frontend rendering loop simple: text events update the body, the single
sources event appends the panel.

**Font choices.** Cinzel (carved-stone Roman, gothic), EB Garamond
(parchment body), JetBrains Mono (data-log feel). All loaded from Google
Fonts. If you want to ship offline, vendor these locally.

**Raven sigil.** Pure SVG built from primitives — no Games Workshop IP.

## Test plan when you re-engage

1. Pull this branch, copy `.env.example` to `.env`, fill in real values.
2. Double-click `launch.bat`. You should see four checkpoints printed to
   the console, then a browser tab opens to `http://127.0.0.1:8000`.
3. Empty state: the motto "Knowledge is Power. Guard it Well." is centered,
   with rotating secondary mottos fading every 5.5s.
4. Ask: *"How does overwatch work?"* — answer streams in, then a "Cited
   Tomes" panel appears below with one or more source chips.
5. Click a source chip — a new tab opens with the PDF at the cited page.
6. Close the launcher console — the server stops.

## Known gotchas

- **Ollama tray app conflict.** If the Ollama tray app is already running,
  `ollama serve` in the launcher will exit immediately because the port is
  bound. That's harmless — the health check on `127.0.0.1:11434` will still
  pass and we move on. Don't treat that as an error in the script.
- **The `model not found` path** is untested in practice because Ollama is
  already set up locally. If it ever runs `ollama pull nomic-embed-text`,
  expect ~250 MB download.
- **PDF page jumps** work in Chrome, Edge, Firefox's built-in viewer. They
  do *not* work if the user has set a third-party PDF viewer as the
  browser's PDF handler (rare but exists).
