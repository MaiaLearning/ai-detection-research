#!/usr/bin/env bash
# Build PDFs from PRACTITIONER_BRIEF.md and TECHNICAL_REPORT.md
#
# Usage:  ./scripts/build_pdf.sh [brief|report|all]
#
# Requires: pandoc, texlive-xetex, lmodern, fonts-crosextra-caladea,
#           fonts-crosextra-carlito, fonts-dejavu
# Debian/Ubuntu:
#   sudo apt-get install -y pandoc texlive-xetex lmodern \
#        fonts-crosextra-caladea fonts-crosextra-carlito fonts-dejavu
# macOS:
#   brew install pandoc && brew install --cask mactex-no-gui
#   Caladea/Carlito: https://fonts.google.com (or substitute — see FONT NOTES)
#
# FONT NOTES
#   Body text is Caladea, headings Carlito. Both are TrueType.
#   Do NOT substitute Bitstream Charter: it exists only as Type 1 on many
#   TeX Live installs, and duplicate Charter faces between
#   /usr/share/fonts/X11/Type1 and texlive-fonts-recommended cause
#   xdvipdfmx to fail with "Error occurred while loading font: bchr8a.pfb".
#   Caladea also lacks Greek and arrows; header.tex maps the four characters
#   the technical report needs (rho, Delta, arrow, geq) to DejaVu Serif.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/.build"
HEADER="$ROOT/scripts/pdf_header.tex"
mkdir -p "$BUILD"

for tool in pandoc xelatex; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found. See header of this script."; exit 1; }
done
[ -f "$HEADER" ] || { echo "ERROR: missing $HEADER"; exit 1; }

# Pull the H1 and byline into pandoc metadata so they render as a title block
# rather than as a section heading, and drop thematic rules (--- ) since
# section rules already separate the document.
prep() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import re, sys
src, out, title, author = sys.argv[1:5]
s = open(src, encoding='utf-8').read()
body = '\n'.join(s.split('\n')[1:])
for lead in ['**Barry Coleman — Chief Technology Officer, MaiaLearning**\n',
             '**Barry Coleman**\nMaiaLearning, Inc.\n`research@maialearning.com`\n']:
    body = body.replace(lead, '', 1)
body = body.replace('[RESOLVED] ', '')          # strip editorial labels if any remain
body = re.sub(r'^---\s*$', '', body, flags=re.M)
body = re.sub(r'\n{3,}', '\n\n', body).strip()
date = __import__('datetime').date.today().strftime('%B %Y')
open(out, 'w', encoding='utf-8').write(
    f'---\ntitle: "{title}"\nauthor: "{author}"\ndate: "{date}"\n---\n\n{body}\n')
PY
}

# --shift-heading-level-by=-1 is REQUIRED: pandoc maps '##' to \subsection,
# and pdf_header.tex styles \section. Without it the styling silently no-ops.
render() {
  pandoc "$1" -o "$2" \
    --pdf-engine=xelatex \
    --shift-heading-level-by=-1 \
    --include-in-header="$HEADER" \
    -V geometry:letterpaper,top=1.05in,bottom=1.05in,left=1.3in,right=1.3in \
    -V mainfont="Caladea" \
    -V sansfont="Carlito" \
    -V monofont="DejaVu Sans Mono" \
    -V fontsize=11pt \
    -V linestretch=1.1 \
    -V colorlinks=true -V linkcolor=linkblue -V urlcolor=linkblue
}

verify() {
  local pdf="$1"
  [ -f "$pdf" ] || { echo "ERROR: $pdf not produced"; exit 1; }
  local pages; pages=$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages/{print $2}' || echo '?')
  local bad=0
  if command -v pdftotext >/dev/null; then
    bad=$(pdftotext "$pdf" - 2>/dev/null | grep -c $'\uFFFD' || true)
  fi
  printf '  %-28s %s pages' "$(basename "$pdf")" "$pages"
  [ "$bad" -gt 0 ] && printf '  ** %s replacement chars — check font coverage **' "$bad"
  printf '\n'
}

build_brief() {
  prep "$ROOT/PRACTITIONER_BRIEF.md" "$BUILD/brief.md" \
    "What We Learned By Shipping an AI Detector for College Essays — And Why We're Withdrawing It" \
    "Barry Coleman — Chief Technology Officer, MaiaLearning"
  render "$BUILD/brief.md" "$ROOT/PRACTITIONER_BRIEF.pdf"
  verify "$ROOT/PRACTITIONER_BRIEF.pdf"
}

build_report() {
  prep "$ROOT/TECHNICAL_REPORT.md" "$BUILD/report.md" \
    "Detection by Uniformity Penalises Good Writing: A Withdrawn Deployment and a Calibration Study on Student Essays" \
    "Barry Coleman — MaiaLearning, Inc. — research@maialearning.com"
  render "$BUILD/report.md" "$ROOT/TECHNICAL_REPORT.pdf"
  verify "$ROOT/TECHNICAL_REPORT.pdf"
}

case "${1:-all}" in
  brief)  build_brief ;;
  report) build_report ;;
  all)    build_brief; build_report ;;
  *)      echo "Usage: $0 [brief|report|all]"; exit 1 ;;
esac

echo "Done."
