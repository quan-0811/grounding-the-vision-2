#!/usr/bin/env bash
set -euo pipefail

ROOT="data/amber"
QUERY_DIR="$ROOT/query"
ZIP_PATH="$ROOT/amber_images.zip"

REPO="https://github.com/junyangwang0410/AMBER"
RAW_BASE="https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data"

# Official Google Drive image file id from AMBER README.
AMBER_IMAGE_FILE_ID="1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"

mkdir -p "$ROOT"
mkdir -p "$QUERY_DIR"

echo "============================================================"
echo "AMBER download"
echo "============================================================"
echo "Repo: $REPO"
echo "Root: $ROOT"
echo

# ------------------------------------------------------------
# Check dependencies
# ------------------------------------------------------------

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python not found."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found. Install curl first."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip not found. Install unzip first."
  exit 1
fi

echo "Installing/upgrading gdown..."
python -m pip install -U -q gdown

# ------------------------------------------------------------
# Download images
# ------------------------------------------------------------

echo
echo "Checking existing AMBER images..."

NUM_IMAGES="$(find "$ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l | tr -d ' ')"

if [ "$NUM_IMAGES" -gt 0 ]; then
  echo "Found $NUM_IMAGES existing image files under $ROOT."
  echo "Skipping image download."
else
  echo "No images found. Downloading AMBER images..."
  gdown --continue \
    "$AMBER_IMAGE_FILE_ID" \
    -O "$ZIP_PATH"

  echo "Extracting images..."
  unzip -q "$ZIP_PATH" -d "$ROOT"

  echo "Removing image zip..."
  rm -f "$ZIP_PATH"
fi

# ------------------------------------------------------------
# Download metadata
# ------------------------------------------------------------

echo
echo "Downloading AMBER metadata files..."

download_file() {
  local url="$1"
  local out="$2"

  echo "  -> $out"
  mkdir -p "$(dirname "$out")"
  curl -fL "$url" -o "$out"
}

download_file "$RAW_BASE/annotations.json" "$ROOT/annotations.json"
download_file "$RAW_BASE/relation.json" "$ROOT/relation.json"
download_file "$RAW_BASE/safe_words.txt" "$ROOT/safe_words.txt"
download_file "$RAW_BASE/metrics.txt" "$ROOT/metrics.txt"

# ------------------------------------------------------------
# Download query files
# ------------------------------------------------------------

echo
echo "Downloading AMBER query files..."

download_file "$RAW_BASE/query/query_generative.json" "$QUERY_DIR/query_generative.json"
download_file "$RAW_BASE/query/query_all.json" "$QUERY_DIR/query_all.json"
download_file "$RAW_BASE/query/query_discriminative.json" "$QUERY_DIR/query_discriminative.json"
download_file "$RAW_BASE/query/query_discriminative-existence.json" "$QUERY_DIR/query_discriminative-existence.json"
download_file "$RAW_BASE/query/query_discriminative-attribute.json" "$QUERY_DIR/query_discriminative-attribute.json"
download_file "$RAW_BASE/query/query_discriminative-relation.json" "$QUERY_DIR/query_discriminative-relation.json"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Done."
echo "============================================================"
echo "AMBER saved under: $ROOT"
echo

echo "Image count:"
find "$ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l

echo
echo "Metadata/query files:"
find "$ROOT" -maxdepth 3 -type f \
  \( -name '*.json' -o -name '*.txt' \) \
  | sort

echo
echo "First few image files:"
find "$ROOT" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
  | sort \
  | head