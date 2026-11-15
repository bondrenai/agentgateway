# Bondren fork of agentgateway

This repository is a fork of [agentgateway](https://github.com/agentgateway/agentgateway). It
exists to build one container image: an upstream release plus the few patches listed below. The
Bondren product runs that image as its inference gateway.

## Why

The product talks to its gateway only in the OpenAI Responses API, without streaming, and sends
each tenant's own provider key with the request. agentgateway v1.7.0 cannot turn a Responses
request into an Anthropic Messages request, so Claude models are unreachable through it. Upstream
has the conversion in review but not merged. Until it ships, this fork carries it.

## Rules

1. The fork is a pinned upstream release plus a short list of patches. Each patch is an upstream
   pull request or is written to become one.
2. No private features. What upstream would not take is done in the gateway's configuration or in
   the product.
3. Every rebase passes the conformance checks before a new image is tagged.
4. When upstream ships an equivalent that passes the same checks, the patch is dropped.

## Layout

| Ref | What it is |
|---|---|
| `main` | A mirror of upstream `main`. It never carries fork commits. |
| `bondren/vX.Y.Z` | Upstream tag `vX.Y.Z` plus the patches and the fork's own files. One branch per upstream release. |
| `bondren-vX.Y.Z-N` | A release tag on that branch. `N` counts fork builds of the same upstream release. Pushing it builds and publishes the image. |

The fork's own files are this directory and `.github/workflows/bondren-image.yml`. Everything
else differs from upstream only by the patches.

## Patches on v1.7.0

List them with `git log --reverse --oneline v1.7.0..bondren/v1.7.0 -- crates`.

| Commit subject | Upstream | Drop when |
|---|---|---|
| Add buffered Responses to Anthropic Messages conversion | Derived from [pull request 2662](https://github.com/agentgateway/agentgateway/pull/2662), cut down to non-streaming requests. | Upstream ships a Responses to Messages conversion that passes the conformance kit. The likely one is the shared conversion layer in [pull request 4383](https://github.com/agentgateway/agentgateway/pull/4383). |
| Keep Anthropic error types when translating errors for OpenAI clients | None yet. It fixes the existing Anthropic to OpenAI error translation, so it can be proposed on its own. | Upstream reports the provider's error type and code itself. |

What the conversion does not do: streaming (a streaming Responses request to an Anthropic
provider is refused, as in upstream v1.7.0), custom and namespaced tools, and the hosted tools.
The product uses none of them.

## Releasing

A release is always: rebase or change, check, tag, record the digest.

### Rebase onto a new upstream release

```sh
git fetch upstream --tags
git switch -c bondren/vX.Y.Z vX.Y.Z
# the commits of the previous branch, oldest first: patches, then the fork's own files
git cherry-pick vA.B.C..bondren/vA.B.C
```

Resolve conflicts as you go. Drop a patch that upstream now covers (rule 4) and delete its row
above. The patches touch little outside one new file,
`crates/llm/src/conversion/messages/from_responses.rs`: the conversion table and three match arms
in `crates/agentgateway/src/llm/mod.rs`, the error translation and a module declaration in
`crates/llm/src/conversion/messages.rs`, the visibility of `ReasoningCarrier` in
`crates/llm/src/conversion/bedrock.rs`, and the golden-test tables.

### Check

1. Upstream's own checks for a pull request, with the toolchain in `rust-toolchain.toml`:

   ```sh
   make lint
   make test
   make generate-schema check-clean-repo check-default-members
   ```

   If a conversion changed on purpose, regenerate the snapshots with
   `INSTA_UPDATE=always cargo test -p agent-llm` and read the diff before committing it.

2. Build the image the way the workflow does and run the conformance kit, both modes
   ([conformance/README.md](conformance/README.md)):

   ```sh
   docker build --build-arg VERSION=X.Y.Z-bondren.N --build-arg GIT_REVISION=$(git rev-parse HEAD) \
     -t agentgateway:candidate .
   python3 bondren/conformance/run.py agentgateway:candidate
   python3 bondren/conformance/run.py agentgateway:candidate --live
   ```

3. In the product repository, point `docker-compose.yml` at the candidate image by digest and run
   `scripts/test-inference-gateway-e2e.sh`. All of its properties must pass.

### Tag and record

```sh
git push origin bondren/vX.Y.Z
git tag -a bondren-vX.Y.Z-N -m "agentgateway vX.Y.Z with the Bondren patch set, build N"
git push origin bondren-vX.Y.Z-N
```

Push only that tag. Never push upstream's `v*` tags to the fork.

The `Bondren image` workflow builds `linux/amd64` and `linux/arm64` on native runners, runs the
stub-mode conformance kit against each, pushes them, publishes the manifest list as
`ghcr.io/bondrenai/agentgateway:bondren-vX.Y.Z-N`, and attests build provenance. Its run summary
prints the digest. That digest is the release: record it in the product change that adopts it.

A tag is never moved or reused. To change a released image, tag the next `N`.

Upstream's own workflows must not run here: they would publish upstream's artifacts under this
organization or wait for runners it does not have. GitHub has not registered any of them in this
fork yet, because the fork's `main` has never been pushed to, so there is nothing to disable and
nothing that can fire. The first push to `main` registers them and starts `Branch` once. Right
after that first sync, cancel that run and disable them:

```sh
for w in release.yml pull_request.yml nightly.yml model-catalog.yml; do
  gh workflow disable "$w" --repo bondrenai/agentgateway
done
```

`Release` (`release.yml`) fires on `v*.*.*` tags and `Branch` (`pull_request.yml`) on pushes to
`main`; `Nightly Release` and `Refresh model catalog` are scheduled. `gh workflow enable` reverses
it. The workflow files themselves stay untouched.

## How the product consumes the image

The product pins the manifest-list digest, never a tag alone:

```yaml
image: ghcr.io/bondrenai/agentgateway:bondren-v1.7.0-1@sha256:<digest from the workflow summary>
```

The package is public, like this repository, so pulling it needs no login. If it is ever made
private, every environment that pulls it needs a token with `read:packages`.

The pin lives in the `inference-gateway` service of the product's `docker-compose.yml` and in its
deployment manifests; both must carry the same digest. The gateway's configuration is the product's
`docker/inference-gateway/config.yaml`.

The product's contract test, `scripts/test-inference-gateway-e2e.sh` (`make
test-inference-gateway-e2e`), reads the image from `docker-compose.yml`, validates that
configuration against it, and drives the product's own client through it. It is the product-side
gate for every digest change; the conformance kit here is the fork-side gate. The product's
runbook, `docs/operations/inference-gateway.md`, has the rest of a digest change, including the
`request_logs` schema fingerprint.

Check the provenance of a digest with
`gh attestation verify oci://ghcr.io/bondrenai/agentgateway@sha256:<digest> --repo bondrenai/agentgateway`.

## What the build pulls from the network

The image is built from the `Dockerfile` unchanged. A build fetches:

| Input | Pinned by |
|---|---|
| `docker.io/library/rust:1.99.0-trixie` (build stage) | Tag. Not a digest. |
| `docker.io/library/node:24.22.1-bookworm` (UI build stage) | Tag. Not a digest. |
| `cgr.dev/chainguard/glibc-dynamic` (runtime base) | Nothing. No tag, so each build takes the current `latest`. |
| `docker.io/docker/dockerfile:1.11` (Dockerfile frontend) | Tag. |
| Rust crates from crates.io | `Cargo.lock`, with checksums (`cargo fetch --locked`). |
| Six Rust git dependencies: forks under `github.com/howardjohn` (`async-openai`, `schemars`, `wiremock-rs`, `yaml-serde-edit`, `rust-tiktoken`) and `gitlab.com/howardjohn/http-serde` | Commit, in `Cargo.toml` and `Cargo.lock`. A deleted fork or commit breaks the build. |
| UI packages from the npm registry, and pnpm itself through corepack | `ui/pnpm-lock.yaml` and the `packageManager` field. |
| GitHub Actions used by the workflow | Commit. |

The three base-image tags are inherited from upstream and can move under the same name, so two
builds of one commit are not guaranteed to be bit-identical. The digest and its attestation
identify what was actually built. The model catalog is compiled in from `catalog/`; the build does
not download it.

## Keeping `main` in sync with upstream

`main` only ever fast-forwards to upstream:

```sh
git fetch upstream
git push origin upstream/main:main
```

GitHub's "Sync fork" button does the same. If the push is rejected, something was committed to
`main` that should not be there; move it to a branch rather than forcing. Do not push tags. The
first sync is also when upstream's workflows have to be disabled (see Releasing).

## When this fork ends

The conversion patch is dropped once an upstream release converts Responses to Anthropic Messages
and passes the conformance kit in both modes. Upstream's shared conversion layer (pull request
4383, off by default while experimental) is the expected route. If no patch is left after a
rebase, the product goes back to upstream's image and this fork is archived.
