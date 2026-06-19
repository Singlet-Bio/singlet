# Singlet MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes the Singlet Atlas as tools for AI assistants.

## Tools

| Tool | Description |
|------|-------------|
| `singlet_stats` | Corpus-wide statistics (total samples, cells, species, success rate) |
| `singlet_search` | Filter samples by organism, protocol, modality, or free text |
| `singlet_qc` | Detailed QC metrics for a specific GSM sample |
| `singlet_load` | Get access info + code snippets to load a sample |
| `singlet_browse` | Paginated sample listing with filters |
| `singlet_protocols` | Protocol distribution and success rates |
| `singlet_quality` | Quality tier breakdown (gold/silver/bronze) |
| `singlet_tissues` | Tissue distribution across samples (37 categories) |
| `singlet_failures` | Failure category breakdown for non-SUCCESS samples |
| `singlet_cell_types` | Cell type distribution (40 normalized categories) |
| `singlet_species` | Species list with sample counts |

## Quick Start

```bash
pip install mcp

# Run smoke test
python -m singlet.mcp.smoke_test

# Start server
python -m singlet.mcp
```

No credentials required. The live tools call the public REST API at
`https://singlet.bio/api` (override with `SINGLET_API_BASE`).

## Configure in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "singlet": {
      "command": "python",
      "args": ["-m", "singlet.mcp"]
    }
  }
}
```

## Configure in VS Code (Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "singlet": {
      "command": "python",
      "args": ["-m", "singlet.mcp"]
    }
  }
}
```

## Configure in Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "singlet": {
      "command": "python",
      "args": ["-m", "singlet.mcp"]
    }
  }
}
```

## Example Interactions

Once configured, you can ask your AI assistant:

- "How many human scRNA-seq samples are in the Singlet Atlas?"
- "Find all 10xv3 mouse brain samples"
- "What are the QC metrics for GSM5238385?"
- "Show me Python code to load GSM5238385"
- "Browse the latest processed samples"

## Architecture

```
┌─────────────────┐    stdio    ┌──────────────────┐
│  Claude/Cursor  │ ◄────────► │  singlet MCP     │
│  VS Code Copilot│            │  server.py       │
└─────────────────┘            └──────────────────┘
                                    │          │
                               ┌────┘          └────┐
                               ▼                    ▼
                     ┌──────────────────┐  ┌──────────────────┐
                     │ Bundled Parquet  │  │ Public REST API  │
                     │ (7 tools, <40ms) │  │ singlet.bio/api  │
                     └──────────────────┘  └──────────────────┘
```

- **7 tools use bundled parquet** (stats, protocols, quality, tissues, failures, cell_types, species) — instant, offline-capable
- **4 tools use the live public REST API** (search, browse, qc, load) — real-time, no auth

Both the MCP server and the website read from the same public API, so data is always consistent.
