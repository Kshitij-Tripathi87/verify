# Demo Security Model

## Scope

This repository is configured as a local, single-user demo. The goal is to prevent common accidental exposure and unsafe ingestion paths during a controlled demonstration. This is not a production security certification.

## Protected paths

| Surface | Demo control | Residual limitation |
|---|---|---|
| Network binding | Loopback default (127.0.0.1) | Operators can override host configuration |
| Browser access | Local CORS allowlist; credentials disabled | CORS is not authentication |
| URL ingestion | HTTP(S), standard ports, public DNS/IP checks, redirect revalidation, 2 MiB response cap, text-only responses | DNS can change between validation and connection; enforce outbound egress rules outside the app |
| File ingestion | Allowlisted extensions, generated storage names, sanitized display names, 5 MiB/file, 10 MiB/request, 10 files/request | Content is not malware-scanned; accept files only from trusted demo participants |
| Answer trust | Citation/source validation + structured claim/code audit | LLM judges are probabilistic; critical technical decisions still require human review |
| Verification outage | Fail closed to safe abstention | The demo may withhold answers when provider credentials or connectivity fail |
| Source archive | Excludes local ChromaDB data, uploaded files, SQLite DBs and private env overrides | Review archive contents before distributing a release |
| Persistence | Local ChromaDB and SQLite | No user authentication, session authorization or tenant isolation |

## Run safely

1. Copy .env.example to .env.
2. Set one supported provider API key. Keep .env out of Git and out of screenshots.
3. Keep HOST set to 127.0.0.1.
4. Start with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000.
5. Confirm GET /health returns a successful liveness response.
6. Confirm GET /ready reports ready. If it reports degraded, check that the seed corpus was indexed and the selected provider key is configured.
7. Test one known-answer question and one out-of-corpus question. The first should return sources only when the evidence auditor approves it; the second should abstain or use graded external evidence.

## Release checklist

- [ ] Confirm the server listens only on 127.0.0.1.
- [ ] Confirm no .env, API key, SQLite database, local ChromaDB directory or uploaded document is present in the source ZIP.
- [ ] Confirm private, loopback, link-local, localhost and non-HTTP URL ingestion attempts are rejected.
- [ ] Confirm redirecting to a private address is rejected.
- [ ] Confirm oversized uploads and remote pages are rejected.
- [ ] Confirm a fabricated citation is rejected.
- [ ] Confirm verifier/provider failure returns a safe abstention, never grounded.
- [ ] Confirm an out-of-corpus query does not receive an unsupported answer.
- [ ] Confirm no public tunnel or shared network exposure is enabled.
- [ ] Run the automated test suite and record the result before the live demo.

## Required before public deployment

Add authentication and authorization, session ownership, tenant isolation, rate limiting, production outbound network egress enforcement, secrets management, persistent job processing, database migrations, structured audit logging, backup/restore, and a deployment-specific threat model.