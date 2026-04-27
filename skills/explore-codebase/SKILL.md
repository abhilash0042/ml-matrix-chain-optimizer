---
name: Explore Codebase
description: Navigate and understand codebase structure
---

## Explore Codebase

Understand the codebase structure using the most efficient tools available.

### Steps

1. If available, use **code-review-graph** MCP tools for high-level structure.
2. Otherwise, use `list_dir` recursively to map the directory tree.
3. Use `grep_search` to find key entry points (e.g., `main`, `app`, `start`).
4. Use `view_file` on configuration files (e.g., `pyproject.toml`, `requirements.txt`, `README.md`) to understand dependencies and purpose.

### Token Efficiency Rules
- Avoid reading entire files if you only need a specific section.
- Use `grep_search` with specific patterns to narrow down your search area.
- Target: complete the initial mapping in ≤5 tool calls.
