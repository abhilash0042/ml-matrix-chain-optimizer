---
name: Debug Issue
description: Systematically debug issues
---

## Debug Issue

Systematically trace and resolve bugs.

### Steps

1. **Locate**: Use `grep_search` or `semantic_search_nodes` (if available) to find relevant code.
2. **Trace**: Map out the data flow or call stack leading to the error.
3. **Reproduce**: Identify the conditions or inputs that trigger the issue.
4. **Fix**: Implement the solution following project standards.
5. **Verify**: Run tests or manual checks to confirm the fix works.

### Token Efficiency Rules
- Use `view_file` with `StartLine`/`EndLine` to focus on the problematic area.
- Do not read unrelated files once the bug area is identified.
