#!/bin/bash
set -e

echo "📦 Downloading test data from Box..."

DATA_DIR="tests/data"
ZIP_FILE="test_data.zip"

# Always remove and recreate data dir to ensure clean state
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# Download zip into current dir (not inside data/)
wget -O "$ZIP_FILE" "https://umd.box.com/shared/static/zd4sai70uw9fs24e1qx6r41ec50pf45g.zip?dl=1"

# Unzip into tests/data directly
unzip "$ZIP_FILE" -d "$DATA_DIR"

# Clean up
rm "$ZIP_FILE"
