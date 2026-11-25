# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.
Use GitHub's private reporting: open the **Security** tab of this repository and choose **Report a vulnerability**.

Include what you found, how to reproduce it and the version or commit you tested.
We will acknowledge the report, work on a fix and tell you when it ships.

## Scope

legal-rag is a library.
It makes no network calls of its own: every model, embedder, reranker, store and cache is supplied by the caller, and the library never reads secrets or settings.
The security-relevant surface is therefore the code that handles untrusted input:

- Document ingestion (parsing of uploaded files, including archives, Office files, email and PDFs).
- The MCP bridge (turning an OpenAPI document into tool calls, and mapping responses back).
- Cache keys and tenancy filters in the pipeline.

A report that a verification check is not a legal opinion is not a vulnerability.
Retrieval and verification results are evidence for a human to review.
