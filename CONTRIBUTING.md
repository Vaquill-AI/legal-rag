# Contributing

Bug reports, fixes and new techniques are welcome.

## Set up

```bash
git clone https://github.com/Vaquill-AI/legal-rag.git
cd legal-rag
python -m venv .venv
.venv/bin/python -m pip install -e ".[all,dev]"
```

Use the virtual environment's `python` for everything, so you test the packages you installed.

## Before you open a pull request

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
```

All three must pass.

## Rules the code follows

- **No hidden configuration.** A component never reads an environment variable, a settings file or a secret. Pass a provider or a value in.
- **Providers are protocols.** Depend on the interfaces in `legal_rag.providers.protocols`. Do not import a vendor SDK outside `providers/adapters` or a lazily imported optional extra.
- **Tests run offline.** Use the fakes in `legal_rag.providers.fakes`. A test never needs a key or a network connection.
- **Rules live in data.** A jurisdiction rule, a prompt or a vocabulary goes in a data file or a plug-in, not in a default code path. `tests/test_repo_hygiene.py` fails on non-US vocabulary in default code.
- **A bug fix ships with a regression test** that fails without the fix.
- **A guide change is tested.** Every Python block in `docs/` runs as written. If you change behaviour a guide describes, update the guide in the same pull request.
- **Say it plainly.** No em dashes in code, comments or docs, and one sentence per line in Markdown.

## Reporting a bug

Open an issue with the smallest input that reproduces it, the output you expected and the output you got.
A failing test is the best report.
For a security problem, follow [SECURITY.md](SECURITY.md).
