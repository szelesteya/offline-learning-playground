# CLAUDE.md

## Math notation

The chat transcript does not render inline `$...$` — it prints the raw dollar
signs (upstream bug; extra braces don't help). But inline `\( ... \)` **does**
render, and so does display `$$...$$`.

- **In chat**: `\( ... \)` for inline math, `$$...$$` for display equations.
  Unicode glyphs (π_θ, π_β, 𝔼, Â, λ, ∇, ≤) are also fine inline. Just never
  inline `$...$`.
- **In files**: normal LaTeX, inline `$...$` included — previewers handle it.
  Don't downgrade files to Unicode.
