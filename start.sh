#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Create .env from example if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — please add your API keys before running again."
  echo "  SERPAPI_KEY    → https://serpapi.com"
  echo "  ANTHROPIC_API_KEY → https://console.anthropic.com"
  exit 1
fi

# Install deps if needed
pip3 install -q -r requirements.txt

echo ""
echo "  Amazon Rank Tracker"
echo "  ─────────────────────────────"
echo "  Open: http://localhost:8000"
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
