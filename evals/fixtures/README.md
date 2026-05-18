# Fixtures

Drop golden conversation fixtures here, one JSON file per case. See
[`../README.md`](../README.md) for the fixture format and the three
planned categories.

Each fixture pins to one or more prompts from
[`app/prompts/registry.py`](../../app/prompts/registry.py) by `(name,
version)`. When a prompt's version bumps, re-run the fixtures that pin
it and refresh their expected outputs.
