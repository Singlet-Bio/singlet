#!/bin/bash
# Convert all notebooks to HTML for website hosting
# Usage: ./build_html.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p html

echo "Converting notebooks to HTML..."
for nb in *.ipynb; do
    name="${nb%.ipynb}"
    echo "  $nb → html/$name.html"
    jupyter nbconvert --to html --output-dir=html --template lab "$nb" 2>/dev/null || echo "    SKIP (invalid)"
done

echo ""
echo "Generated $(ls html/*.html 2>/dev/null | wc -l) HTML files:"
ls -lh html/*.html | awk '{print "  " $NF " (" $5 ")"}'
