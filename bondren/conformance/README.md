# Conformance kit

Checks that a gateway image still does what the Bondren product relies on. Every image must pass
it before it is tagged (rule 3 in [../README.md](../README.md)); the image workflow runs the stub
mode on each architecture and pushes nothing if it fails.

It needs Python 3.9 or later and Docker. Nothing else is installed.

## Stub mode

```sh
python3 bondren/conformance/run.py <image>
```

Starts the image and a Postgres for its request log in Docker, with `stub.py` standing in for
OpenAI, Anthropic and Gemini. No provider is contacted and no real key is involved. It takes a
few seconds. The exit status is non-zero if any check fails.

For each provider entry it sends the request shapes the product sends (`fixtures/`, mirrored from
the product's `internal/inferencewire/wire.go`) and asserts:

- the upstream request the gateway built: plain text, a forced strict tool, `text.format`, a tool
  loop that replays reasoning, reasoning effort, `service_tier: flex`, and an image;
- the reply, read the way the product reads it (`product.py` is as strict as the product's parser);
- the `request_logs` row: provider, model, token counts including cache reads and writes, the
  attribution columns, and the cost at rates pinned in `run.py`;
- that the caller's key is forwarded as the provider expects (`x-api-key` for Anthropic) and is
  not written to the log store or the gateway's output;
- error shapes: status, `retry-after`, and the error type, code and message.

A `GAP` line is a known upstream limitation that the fork does not change. It is reported, not
failed, and the check will start passing by itself if upstream fixes it. Today there is one: the
Responses to chat-completions conversion drops `input_image` parts, so an image never reaches a
Gemini model.

`--only TEXT` runs the checks whose name contains `TEXT`. `--keep` leaves the containers running
for inspection. `--platform linux/amd64` runs an image of another architecture under emulation.
`--record DIR` writes every gateway reply to `DIR`, for feeding the product's real parser.
`CONFORMANCE_POSTGRES_IMAGE` replaces the Postgres image (default `pgvector/pgvector:pg17`, the one
the product runs).

## Live mode

```sh
ANTHROPIC_API_KEY_FILE=/path/to/anthropic-key \
GEMINI_API_KEY_FILE=/path/to/gemini-key \
python3 bondren/conformance/run.py <image> --live
```

Repeats the checks a stub cannot prove against the real APIs: Claude accepts replayed signed
thinking, structured output through `text.format`, an image, the prompt-caching policy with its
cache-write and cache-read rows, real error bodies; Gemini accepts a tool loop and `text.format`.
It costs a few cents.

Each variable names a file that holds one key. The key is read inside the process, sent only as
the request credential, and removed from anything the script prints; it is never passed on a
command line or written to disk. `--only claude` or `--only gemini` needs only that provider's
key file.

The models are `claude-haiku-5-5`, `claude-sonnet-5-5` for the second structured-output check,
and `gemini-3.5-flash-lite`; `CONFORMANCE_CLAUDE_MODEL`, `CONFORMANCE_CLAUDE_STRUCTURED_MODEL` and
`CONFORMANCE_GEMINI_MODEL` replace them.

## Changing it

Add a request shape as a fixture when the product starts sending one, and a check that states
what must reach the provider and what must come back. When the product's parser changes, change
`product.py` with it. The product's own contract test (`scripts/test-inference-gateway-e2e.sh` in
the Bondren repository) stays the authority for the OpenAI routes; this kit covers the providers
that test does not reach.
