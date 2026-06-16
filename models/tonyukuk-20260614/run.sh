#!/usr/bin/env bash
set -e
uv run orkhon generate --checkpoint models/tonyukuk-20260614/checkpoint --tokenizer models/tonyukuk-20260614/tokenizer -p "Once upon a time"
