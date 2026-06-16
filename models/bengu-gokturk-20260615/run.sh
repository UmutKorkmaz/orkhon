#!/usr/bin/env bash
set -e
uv run orkhon chat --checkpoint models/bengu-gokturk-20260615/checkpoint --tokenizer models/bengu-gokturk-20260615/tokenizer
