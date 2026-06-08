#!/usr/bin/env bash
# Neutrality check: fail if the repo contains any product/source-revealing terms.
#
# The forbidden-term list is intentionally NOT stored in this repository (that
# would leak the very thing it guards against). Keep the patterns in a file
# outside the repo and pass its path via HYGIENE_PATTERNS. The file holds one
# regex fragment per line; matching is case-insensitive.
set -euo pipefail

patterns_file="${HYGIENE_PATTERNS:-}"
if [[ -z "$patterns_file" || ! -f "$patterns_file" ]]; then
  echo "Set HYGIENE_PATTERNS to a patterns file kept outside this repo." >&2
  exit 2
fi

if grep -rIinf "$patterns_file" --exclude-dir=.git --exclude-dir=.venv . ; then
  echo "Found terms that should not appear in a neutral public repo. Remove them." >&2
  exit 1
fi
echo "clean"
