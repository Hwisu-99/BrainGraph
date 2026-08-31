# AutoNote (BrainGraph)

Turn what you read into a knowledge graph you actually own, then talk to it.

AutoNote is the implementation of **BrainGraph**: a personal, self-growing knowledge graph (a "Brain") built out of papers you feed it. An LLM drafts the first version of the graph automatically, but the graph is meant to be *yours* — you edit it by hand until it precisely reflects your own understanding, then connect it back to an LLM so the knowledge you've accumulated actually improves the answers you get.

## Core ideas

- **Brain** — the knowledge graph you're building: `concept`/`entity` nodes connected by typed relations, optionally grouped into multiple named containers (e.g. "Robot Brain", "RL Brain").
- **One node = one Markdown file** — every concept or entity is a single `.md` file holding its own description, aliases, categories, and the papers it came from. Nothing lives only inside a database you can't read.
- **Base graph, then refinement** — the first pass comes from an LLM reading your source material; from there you create, delete, merge, and re-link nodes by hand (via the graph UI or through conversation) until the graph says what you actually know.

## What's implemented today

| Feature | Status |
|---|---|
| PDF → structured Obsidian note + concept/entity graph (Claude extraction) | done |
| Manual graph editing (create/delete/merge nodes, link/unlink papers & concepts) | done |
| Typed semantic relations between concepts/entities (`USES`, `EXTENDS`, `PART_OF`, `COMPARED_TO`, ...) with a dedicated Semantic View | done |
| Neo4j mirror + hybrid search for GraphRAG | done (Claude only, via MCP) |
| Self-improving loop — Claude can create/link nodes mid-conversation | done (MCP tools) |
| Multiple named Brains (containers) + merging two Brains | done |
| GPT / Gemini GraphRAG connectors | planned |
| Brain consolidation (importing someone else's Brain) | planned |
| Schema that auto-adapts to non-paper domains | planned |

## How it works

1. Drop a PDF into the web UI.
2. The backend extracts the text (PyMuPDF) and sends it to Claude with a structured JSON schema — a narrative summary plus the key concepts/entities and the semantic relations between them.
3. Claude's response becomes:
   - `<vault>/AutoNote/<slug>/<slug>.md` — the note itself, wikilinked to its concepts
   - `<vault>/AutoNote/<slug>/<slug>.excalidraw` — a concept diagram
   - one `.md` node file per new/updated concept and entity (`_concepts/`, `_entities/`), each carrying its own relations
4. The note is uploaded to Supabase Storage; concept/entity nodes and relations are mirrored into Neo4j.
5. The web UI gives you:
   - A force-directed **graph view** — papers, tags, and concepts, focusable per paper
   - A **Semantic View** for the typed concept/entity relation graph (hover an edge to see its type, rationale, and source papers)
   - The estimated USD cost of each Claude call, computed from real token usage for whichever model served the request
6. `mcp_server.py` exposes the same Brain (search, node CRUD, linking) to Claude Desktop/Code over MCP, so you can query and grow the graph from an ordinary conversation — web-UI edits and MCP edits go through the same FastAPI backend, so the two never drift apart.

## Requirements

- Python 3.11+
- An Obsidian vault on disk
- An [Anthropic API key](https://console.anthropic.com/)
- A [Supabase](https://supabase.com/) project with a Storage bucket
- A [Neo4j](https://neo4j.com/) instance (the Aura free tier works) for GraphRAG/MCP search

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Fill in `.env`:

```
ANTHROPIC_API_KEY=your-anthropic-api-key
OBSIDIAN_VAULT_PATH=C:\path\to\your\Obsidian\vault
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key
SUPABASE_BUCKET=autonote-notes
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

`SUPABASE_KEY` should be a `service_role` key since notes are uploaded from trusted server-side code — never expose this key client-side. `NEO4J_*` is optional: without it everything still works except Neo4j sync and GraphRAG search (those failures are caught and logged, not fatal).

## Run

```bash
uvicorn app:app --reload --port 8123
```

Open `http://localhost:8123`.

To let Claude Desktop/Code query and grow the Brain directly, register `mcp_server.py` as an MCP server (see `.mcp.json`) — it talks to the same running `app.py` over HTTP, so start `uvicorn` first.

## Two views into the Brain, both editable

- **Graph view** (`/`) — nodes plus `LINKED_TO` edges only: which paper each concept/entity came from, and which entities sit under which concept. This is where you create, delete, or merge concept/entity nodes, and link or unlink them to papers.
- **Semantic View** (`/semantic_view.html`) — the same nodes, plus every typed semantic relation between them (`USES`, `EXTENDS`, `PART_OF`, `COMPARED_TO`, ...). Drag between two nodes to create a new relation, hover an edge to see its type/rationale/source papers, and edit or delete existing relations right on the graph.

Neither view is read-only — what Claude extracts from a paper is only a first draft. You're expected to reshape nodes, edges, and relations by hand in whichever view until the graph matches what you actually know.

## Notes on cost

Claude API pricing is looked up per-model in `paper_notes/claude_client.py` (`_PRICE_PER_MTOK`), keyed by the exact model ID that served the response — so cost reporting stays accurate even if the configured model changes. If you switch to a model not yet in that table, adding it there is required (a `ValueError` is raised instead of silently reporting the wrong price).
