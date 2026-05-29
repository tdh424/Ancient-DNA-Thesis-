#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="${1:-$ROOT_DIR/Package}"
OVERRIDES_DIR="$ROOT_DIR/overrides"

if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "Package directory not found: $PACKAGE_DIR" >&2
  exit 1
fi

copy_override() {
  local rel="$1"
  local src="$OVERRIDES_DIR/$rel"
  local dst="$PACKAGE_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    echo "Missing override file: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

require_path() {
  local rel="$1"
  local dst="$PACKAGE_DIR/$rel"
  if [[ ! -e "$dst" ]]; then
    echo "Expected path missing in Package: $dst" >&2
    exit 1
  fi
}

replace_gzstream_include() {
  local rel="$1"
  local dst="$PACKAGE_DIR/$rel"
  require_path "$rel"
  perl -0pi -e 's/#include <gzstream\.h>/#include "gzstream\/gzstream.h"/g' "$dst"
}

copy_override "Makefile"
copy_override "gargammel.pl"
copy_override "src/Makefile"

require_path "libgab"
copy_override "libgab/Makefile"
copy_override "libgab/gzstream/version.txt"
rm -f "$PACKAGE_DIR/libgab/gzstream/version"

for rel in \
  "src/fragSim.cpp" \
  "src/deamSim.cpp" \
  "src/adptSim.cpp" \
  "src/fasta2fastas.cpp" \
  "src/damage_patterns2prof.cpp" \
  "src/mapDamage2prof.cpp" \
  "libgab/libgab.h" \
  "libgab/FastQParser.h"
  do
  replace_gzstream_include "$rel"
done

echo "Applied local Gargammel overrides to: $PACKAGE_DIR"
