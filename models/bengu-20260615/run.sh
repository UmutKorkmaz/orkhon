#!/usr/bin/env bash
set -e
uv run orkhon generate --checkpoint models/bengu-20260615/checkpoint --tokenizer models/bengu-20260615/tokenizer -p "Once upon a time"
