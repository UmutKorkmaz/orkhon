#!/usr/bin/env bash
set -e
uv run orkhon chat --checkpoint models/bumin-mini-20260620/checkpoint --tokenizer models/bumin-mini-20260620/tokenizer
