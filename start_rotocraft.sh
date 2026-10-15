#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "======================================================="
echo "  Rotocraft Studio: AI Rotoscoping for FilmCraft"
echo "======================================================="
echo ""

PYTHON_EXE=""

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f "../../tools/rotocraft/.venv/bin/python" ]; then
    PYTHON_EXE="../../tools/rotocraft/.venv/bin/python"
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[Rotocraft] Setting up environment for first run..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv
        uv pip install --python .venv/bin/python -e .
        PYTHON_EXE=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
        .venv/bin/python -m pip install -e .
        PYTHON_EXE=".venv/bin/python"
    else
        echo "[ERROR] Python 3.10+ was not found on your system."
        exit 1
    fi
fi

echo "[Rotocraft] Starting Rotocraft Studio on http://127.0.0.1:8765..."
echo "Press Ctrl+C to stop."
echo ""

"$PYTHON_EXE" -m rotocraft.cli app --port 8765
