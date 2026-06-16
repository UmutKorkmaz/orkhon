# Security Policy

## Supported versions

Only the latest `main` branch is supported with security fixes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Until a GitHub Security Advisory
is configured, report privately to the maintainer. Include: a minimal repro, the affected file,
and the impact.

## Threat model & boundaries

Orkhon is a **local-first research/education stack**, not a hardened multi-tenant service.
The boundaries below are the ones to keep in mind:

| Surface | Status | Do NOT trust it with |
|---|---|---|
| **MBPP sandbox** (`eval/code_sandbox.py`) | **Trusted-local only.** It blocks obvious `os`/`socket`/`open` calls via an import allowlist + rlimits, but is **bypassable** by object-graph tricks. | Untrusted model-generated code. For that, use a real container (nsjail/firejail/seccomp), no network, read-only FS, low-priv user. |
| **`read_file` tool** | Jailed to explicit `--file-root`s (never the cwd by default). | Reading files outside an explicitly-granted root. |
| **`/v1/agent/run` HTTP** | Escapes all external message content (neutralizes `<\|end\|>`/`<\|system\|>` injection); rejects inbound `role="tool"`. | Exposure on the public internet without auth + a reverse proxy. Bind to `127.0.0.1` (the default). |
| **RAG retrieved text** | Treated as untrusted data (escaped before re-encoding). | A malicious indexed document is still data; keep any future code-execution tool separate from untrusted indexes. |

## Hard rules for deployments

1. Never expose `orkhon serve` directly to the internet without authentication.
2. Do not add public code execution without real container isolation.
3. Treat any tool/RAG index content as adversarial input.
