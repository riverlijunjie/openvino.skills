#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci/private_builds/dldt/releases/2026/1/commit/releases/2026/1/private_linux_ubuntu_22_04_release/cpack/}"
OUT_DIR="${2:-./cpack_downloads}"

mkdir -p "$OUT_DIR"

TMP_HTML="$(mktemp)"
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_HTML" "$TMP_LIST"' EXIT

printf 'Fetching directory listing...\n'
curl -fsSL "$BASE_URL" -o "$TMP_HTML"

python3 - "$BASE_URL" "$TMP_HTML" > "$TMP_LIST" <<'PY'
import sys
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

base_url = sys.argv[1]
html_path = sys.argv[2]

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a':
            return
        for key, value in attrs:
            if key.lower() == 'href' and value:
                self.links.append(value)

with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
    html = f.read()

parser = LinkParser()
parser.feed(html)

seen = set()
base_path = urlparse(base_url).path

for href in parser.links:
    if href in ('../', './', '/'):
        continue
    if href.startswith('?') or href.startswith('#'):
        continue

    full_url = urljoin(base_url, href)
    parsed = urlparse(full_url)

    if parsed.path.endswith('/'):
        continue
    if not parsed.path.startswith(base_path):
        continue
    if full_url in seen:
        continue

    seen.add(full_url)
    print(full_url)
PY

FILE_COUNT="$(wc -l < "$TMP_LIST" | tr -d ' ')"
if [[ "$FILE_COUNT" == "0" ]]; then
    echo "No files found under: $BASE_URL" >&2
    exit 1
fi

printf 'Found %s files. Downloading to %s\n' "$FILE_COUNT" "$OUT_DIR"

while IFS= read -r file_url; do
    [[ -z "$file_url" ]] && continue
    file_name="${file_url##*/}"
    printf 'Downloading %s\n' "$file_name"
    curl -fL --retry 3 --retry-delay 2 -o "$OUT_DIR/$file_name" "$file_url"
done < "$TMP_LIST"

echo "Done. Files saved in: $OUT_DIR"
