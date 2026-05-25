#!/bin/bash
# US Stocks Screener - Hourly Runner
# Set your Telegram credentials before running:
#   export TELEGRAM_BOT_TOKEN="your_bot_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

# Load credentials from .env if present
if [ -f "$DIR/.env" ]; then
    set -a
    source "$DIR/.env"
    set +a
fi

python3 daily_screener.py
