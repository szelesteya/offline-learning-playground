#!/usr/bin/env bash
# render_mermaid.sh — convert a Mermaid .mmd source file to an SVG next to it.
#
# Usage:
#   bash render_mermaid.sh path/to/diagram.mmd
#
# Produces path/to/diagram.svg. Intended for Marp decks, where Mermaid is not
# native. The .mmd source is kept alongside the .svg so diagram edits stay
# diff-friendly.
#
# Requires: Node.js + npx. Uses @mermaid-js/mermaid-cli at its current major
# (pin to 11.x for reproducibility).

set -euo pipefail

MERMAID_CLI_VERSION="${MERMAID_CLI_VERSION:-11}"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <path-to-diagram.mmd>" >&2
  exit 64
fi

input="$1"

if [[ ! -f "$input" ]]; then
  echo "error: input file not found: $input" >&2
  exit 66
fi

if [[ "${input##*.}" != "mmd" ]]; then
  echo "error: input must have a .mmd extension: $input" >&2
  exit 65
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "error: npx is required (install Node.js)" >&2
  exit 69
fi

output="${input%.mmd}.svg"

npx --yes "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
  --input "$input" \
  --output "$output" \
  --backgroundColor transparent

echo "rendered: $output"
