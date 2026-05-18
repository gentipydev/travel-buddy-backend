# Runners

Where the eval harness implementations live. Empty for now; the
recommended starting point is a thin Python wrapper around
[Promptfoo](https://promptfoo.dev) or
[Anthropic Evals](https://docs.anthropic.com/en/docs/test-and-evaluate)
that loads fixtures from `../fixtures/` and writes a scorecard JSON.

The runner should record cache-hit rate (`cache_read_input_tokens /
total input tokens`) from `response.usage` so the eval also surfaces a
caching-regression signal.
