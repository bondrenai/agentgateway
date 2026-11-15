#!/usr/bin/env python3
"""Conformance checks for a gateway image.

    python3 run.py IMAGE            # stub mode: no provider is contacted
    python3 run.py IMAGE --live     # live mode: the key checks against the real providers

Starts the image in Docker with a Postgres for its request log and drives it with the request
shapes the product sends (fixtures/). In stub mode the providers are stub.py, so each check can
assert the upstream request the gateway built, the reply the product parses, the request_logs row
and its cost, key forwarding, and error shapes. Needs only Python 3.9+ and Docker.

Live mode reads provider keys from the files named by ANTHROPIC_API_KEY_FILE and
GEMINI_API_KEY_FILE. A key is held in this process only for the length of the run, is sent only
as the request credential, and is removed from anything this script prints.
"""
import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

import product
import stub as stubs

HERE = os.path.dirname(os.path.abspath(__file__))
POSTGRES_IMAGE = os.environ.get("CONFORMANCE_POSTGRES_IMAGE", "pgvector/pgvector:pg17")
CLAUDE_MODEL = os.environ.get("CONFORMANCE_CLAUDE_MODEL", "claude-haiku-5-5")
# Checked on a second Claude model because it is the one the product reaches structured output
# on through text.format rather than a forced tool.
CLAUDE_STRUCTURED_MODEL = os.environ.get("CONFORMANCE_CLAUDE_STRUCTURED_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.environ.get("CONFORMANCE_GEMINI_MODEL", "gemini-3.5-flash-lite")

# Stub-mode models. Their rates and thinking tags are pinned in CATALOG so the expected cost of a
# request does not move when upstream refreshes its built-in catalog. Rates are per million tokens.
RATES = {"input": "1", "output": "5", "cacheRead": "0.1", "cacheWrite": "1.25", "cacheWriteDetails": {"1h": "2"}}
CATALOG = {"providers": {
    "openai": {"models": {
        "gpt-conformance": {"rates": RATES},
        "text-embedding-conformance": {"rates": {"input": "1"}},
    }},
    "anthropic": {"models": {
        "claude-conformance": {"rates": RATES, "tags": ["adaptive_thinking"]},
        "claude-conformance-cached": {"rates": RATES, "tags": ["adaptive_thinking"]},
        "claude-conformance-legacy": {"rates": RATES, "tags": ["legacy_thinking"]},
    }},
    "gcp.gemini": {"models": {"gemini-conformance": {"rates": RATES}}},
}}

CONFIG = """\
config:
  modelCatalog:
    - file: /cfg/catalog.json
  database:
    url: postgres://agw:agw@{postgres}:5432/agw
  standardAttributes:
    user:  'request.headers["x-bondren-tenant"]'
    group: 'request.headers["x-bondren-purpose"]'
llm:
  port: 4000
  models:
    - name: "gpt-conformance"
      provider: openai
      params: {{ baseUrl: "{stub}/openai/v1" }}
    - name: "text-embedding-conformance"
      provider: openai
      params: {{ baseUrl: "{stub}/openai/v1" }}
    - name: "claude-conformance"
      provider: anthropic
      params: {{ baseUrl: "{stub}/anthropic/v1" }}
    - name: "claude-conformance-legacy"
      provider: anthropic
      params: {{ baseUrl: "{stub}/anthropic/v1" }}
    - name: "claude-conformance-cached"
      provider: anthropic
      params: {{ baseUrl: "{stub}/anthropic/v1" }}
      promptCaching: {{}}
    - name: "gemini-conformance"
      provider: gemini
      params: {{ baseUrl: "{stub}/gemini" }}
    # Live mode. Selected by header so a stub-mode run can never reach a provider.
    - name: "claude-*"
      provider: anthropic
      matches:
        - headers:
            - name: x-bondren-route
              value: {{ exact: live }}
    - name: "claude-*"
      provider: anthropic
      promptCaching: {{}}
      matches:
        - headers:
            - name: x-bondren-route
              value: {{ exact: live-cached }}
    - name: "gemini-*"
      provider: gemini
      matches:
        - headers:
            - name: x-bondren-route
              value: {{ exact: live }}
frontendPolicies:
  accessLog:
    database:
      add:
        bondren_feature: 'request.headers["x-bondren-feature"]'
        bondren_mode: 'request.headers["x-bondren-mode"]'
        bondren_route: 'request.headers["x-bondren-route"]'
"""

ROW_COLUMNS = (
    "http_status, gen_ai_provider_name, gen_ai_request_model, gen_ai_response_model, input_tokens, "
    "output_tokens, total_tokens, cache_read_input_tokens, cache_creation_input_tokens, "
    "reasoning_tokens, cost, agentgateway_user, agentgateway_group, attributes_json"
)

SECRETS = []


def scrub(text):
    """Remove provider keys from text before it is printed."""
    for secret in SECRETS:
        text = text.replace(secret, "<redacted>")
    return text


class Failed(Exception):
    pass


class Gap(Exception):
    """A known upstream limitation: reported, not counted as a failure."""


def expect(condition, message):
    if not condition:
        raise Failed(message)


def same(actual, expected, what):
    if actual != expected:
        raise Failed("%s: got %s, want %s" % (what, json.dumps(actual)[:600], json.dumps(expected)[:600]))


def fixture(name, model, **changes):
    with open(os.path.join(HERE, "fixtures", name + ".json")) as f:
        body = json.load(f)
    body["model"] = model
    body.update(changes)
    return body


def docker(*args, check=True, capture=True):
    result = subprocess.run(("docker",) + args, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None, text=True)
    if check and result.returncode != 0:
        raise RuntimeError("docker %s failed: %s" % (args[0], scrub(result.stdout or "")[-2000:]))
    return (result.stdout or "").strip()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Reply:
    def __init__(self, status, headers, raw, tenant):
        self.status, self.headers, self.raw, self.tenant = status, headers, raw, tenant

    def parsed(self):
        expect(self.status == 200, "gateway answered %d: %s" % (self.status, scrub(self.raw.decode("utf-8", "replace"))[:600]))
        try:
            return product.parse_reply(self.raw)
        except product.ContractError as err:
            raise Failed("the product would reject this reply (%s): %s" % (err, scrub(self.raw.decode("utf-8", "replace"))[:900]))

    def error(self):
        return product.error_detail(self.raw)


class Lab:
    """The gateway under test with its request-log database, and the stub the stub-mode routes use."""

    def __init__(self, image, platform_arg=None, keep=False, record=None):
        self.image, self.platform, self.keep, self.record = image, platform_arg, keep, record
        self.check = ""
        self.calls = 0
        self.id = uuid.uuid4().hex[:8]
        self.network = "agw-conformance-" + self.id
        self.postgres = self.network + "-pg"
        self.gateway = self.network + "-gw"
        self.stub = stubs.Stub()
        self.tmp = None
        self.port = None

    def __enter__(self):
        # Containers reach the host's loopback on Docker Desktop, the bridge address on Linux.
        stub_port = self.stub.start("127.0.0.1" if platform.system() == "Darwin" else "0.0.0.0")
        self.tmp = tempfile.TemporaryDirectory(prefix="agw-conformance-")
        os.chmod(self.tmp.name, 0o755)
        config = CONFIG.format(postgres=self.postgres, stub="http://host.docker.internal:%d" % stub_port)
        for name, content in (("config.yaml", config), ("catalog.json", json.dumps(CATALOG))):
            path = os.path.join(self.tmp.name, name)
            with open(path, "w") as f:
                f.write(content)
            os.chmod(path, 0o644)
        docker("network", "create", self.network)
        docker("run", "-d", "--name", self.postgres, "--network", self.network,
               "-e", "POSTGRES_USER=agw", "-e", "POSTGRES_PASSWORD=agw", "-e", "POSTGRES_DB=agw", POSTGRES_IMAGE)
        self._wait("postgres", lambda: "accepting" in docker(
            "exec", self.postgres, "pg_isready", "-U", "agw", "-d", "agw", check=False))
        self.port = free_port()
        run = ["run", "-d", "--name", self.gateway, "--network", self.network,
               "--add-host", "host.docker.internal:host-gateway",
               "-p", "127.0.0.1:%d:4000" % self.port, "-v", self.tmp.name + ":/cfg:ro"]
        if self.platform:
            run += ["--platform", self.platform]
        docker(*run, self.image, "-f", "/cfg/config.yaml")
        self._wait("gateway", self._listening)
        return self

    def __exit__(self, *exc):
        self.stub.stop()
        if self.keep:
            print("kept: containers %s %s, network %s" % (self.gateway, self.postgres, self.network))
            return
        docker("rm", "-f", "-v", self.gateway, self.postgres, check=False)
        docker("network", "rm", self.network, check=False)
        self.tmp.cleanup()

    def _wait(self, what, ready, seconds=90):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if ready():
                return
            time.sleep(0.5)
        raise RuntimeError("%s did not become ready:\n%s" % (what, self.logs()))

    def _listening(self):
        if "true" not in docker("inspect", "-f", "{{.State.Running}}", self.gateway, check=False):
            raise RuntimeError("the gateway container exited:\n" + self.logs())
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/" % self.port, timeout=2)
        except urllib.error.HTTPError:
            return True
        except OSError:
            return False
        return True

    def logs(self, tail=25):
        return scrub(docker("logs", "--tail", str(tail), self.gateway, check=False))

    def call(self, body, key, route="stub", path="/v1/responses", purpose="assist"):
        """One product request: the caller's provider key as the bearer credential, plus the
        attribution headers the product sets. Each call gets its own synthetic tenant."""
        tenant = str(uuid.uuid4())
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path), method="POST",
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                     "x-bondren-tenant": tenant, "x-bondren-purpose": purpose,
                     "x-bondren-feature": "conformance", "x-bondren-mode": "byo", "x-bondren-route": route})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = Reply(response.status, dict(response.headers), response.read(), tenant)
        except urllib.error.HTTPError as err:
            reply = Reply(err.code, dict(err.headers), err.read(), tenant)
        if self.record:
            self.calls += 1
            name = "%s-%02d.response.json" % (self.check, self.calls)
            with open(os.path.join(self.record, name), "w") as f:
                json.dump({"status": reply.status, "body": reply.raw.decode("utf-8", "replace")}, f, indent=1)
        return reply

    def sql(self, statement):
        return docker("exec", self.postgres, "psql", "-U", "agw", "-d", "agw", "-tAc", statement)

    def row(self, reply):
        """The request_logs row the gateway wrote for a call. Rows are written after the reply."""
        statement = ("SELECT row_to_json(t) FROM (SELECT %s FROM request_logs WHERE agentgateway_user = '%s' "
                     "ORDER BY completed_at DESC LIMIT 1) t" % (ROW_COLUMNS, uuid.UUID(reply.tenant)))
        deadline = time.time() + 15
        while time.time() < deadline:
            out = self.sql(statement)
            if out.startswith("{"):
                return json.loads(out)
            time.sleep(0.3)
        raise Failed("no request_logs row was written for the call")


def price(usage):
    """Cost of a request at the pinned stub-mode rates, from token counts by billing class."""
    per_token = {"input": RATES["input"], "output": RATES["output"], "cache_read": RATES["cacheRead"],
                 "cache_write": RATES["cacheWrite"], "cache_write_1h": RATES["cacheWriteDetails"]["1h"]}
    return sum(tokens * float(per_token[kind]) for kind, tokens in usage.items()) / 1e6


def check_row(lab, reply, provider, model, tokens, cost, route="stub"):
    row = lab.row(reply)
    same(row["http_status"], 200, "request_logs.http_status")
    same(row["gen_ai_provider_name"], provider, "request_logs.gen_ai_provider_name")
    same(row["gen_ai_request_model"], model, "request_logs.gen_ai_request_model")
    for column, value in tokens.items():
        same(row[column], value, "request_logs." + column)
    expect(row["cost"] is not None and abs(row["cost"] - cost) < 1e-9,
           "request_logs.cost: got %r, want %.6f" % (row["cost"], cost))
    same(row["agentgateway_user"], reply.tenant, "request_logs.agentgateway_user")
    same(row["agentgateway_group"], "assist", "request_logs.agentgateway_group")
    attributes = row["attributes_json"]
    for key, value in (("bondren_feature", "conformance"), ("bondren_mode", "byo"), ("bondren_route", route)):
        same(attributes.get(key), value, "request_logs.attributes_json." + key)
    return row


def upstream(lab, path, count=1):
    """The requests the stub received for the checks' last calls, all on the expected endpoint."""
    requests = lab.stub.take()
    same([r["path"] for r in requests], [path] * count, "upstream requests")
    return requests if count != 1 else requests[0]


# ---- stub mode ---------------------------------------------------------------------------------

STUB_CHECKS = []
LIVE_CHECKS = []


def stub_check(fn):
    STUB_CHECKS.append(fn)
    return fn


def live_check(fn):
    LIVE_CHECKS.append(fn)
    return fn


OPENAI = "/openai/v1/responses"
ANTHROPIC = "/anthropic/v1/messages"
GEMINI = "/gemini/v1beta/openai/chat/completions"
TOOL_SCHEMA = {"type": "object", "properties": {"echo": {"type": "string"}}, "required": ["echo"],
               "additionalProperties": False}
OPENAI_TOKENS = {"input_tokens": 8120, "output_tokens": 400, "total_tokens": 8520,
                 "cache_read_input_tokens": 5000, "reasoning_tokens": 340}
OPENAI_COST = price({"input": 8120 - 5000, "cache_read": 5000, "output": 400})


@stub_check
def openai_passes_requests_through(lab):
    """OpenAI entries are a passthrough: every product shape reaches the provider unchanged."""
    for name in ("text", "forced-strict-tool", "background", "text-format", "image"):
        body = fixture(name, "gpt-conformance")
        reply = lab.call(body, "openai-key-" + name)
        sent = upstream(lab, OPENAI)
        same(sent["body"], body, "upstream body for " + name)
        same(sent["headers"].get("authorization"), "Bearer openai-key-" + name, "upstream authorization")
        expect("x-api-key" not in sent["headers"], "x-api-key must not be sent to OpenAI")
        parsed = reply.parsed()
        if name == "text":
            same(parsed["text"], stubs.TEXT_MARKER, "reply text")
            same(parsed["usage"], {"input_tokens": 8120, "output_tokens": 400, "cached_tokens": 5000}, "reply usage")
            check_row(lab, reply, "openai", "gpt-conformance", OPENAI_TOKENS, OPENAI_COST)
        if name == "forced-strict-tool":
            same(parsed["calls"], [{"id": "call_conformance", "name": "probe_tool", "arguments": {"echo": stubs.TOOL_MARKER}}], "tool call")


@stub_check
def openai_tool_loop_replays_reasoning(lab):
    first_body = fixture("forced-strict-tool", "gpt-conformance")
    first = lab.call(first_body, "openai-key").parsed()
    upstream(lab, OPENAI)
    second_body = product.replay_input(first_body, first, "Invalid input: echo must be the word pineapple.")
    second = lab.call(second_body, "openai-key").parsed()
    sent = upstream(lab, OPENAI)
    same(sent["body"]["input"], second_body["input"], "replayed input upstream")
    expect(any(item.get("encrypted_content") == stubs.OPENAI_REASONING for item in sent["body"]["input"]),
           "encrypted reasoning did not reach the provider")
    same(second["calls"][0]["arguments"], {"echo": stubs.CORRECTED_MARKER}, "corrected tool call")


@stub_check
def openai_embeddings(lab):
    body = fixture("embeddings", "text-embedding-conformance")
    reply = lab.call(body, "openai-key", path="/v1/embeddings", purpose="embed")
    sent = upstream(lab, "/openai/v1/embeddings")
    same(sent["body"], body, "upstream embeddings body")
    same(sent["headers"].get("authorization"), "Bearer openai-key", "upstream authorization")
    expect(reply.status == 200, "embeddings answered %d" % reply.status)
    vector = json.loads(reply.raw)["data"][0]["embedding"]
    expect(len(vector) == 1536 and vector[0] == 0.25 and vector[-1] == 0.25, "embedding did not round-trip")
    row = lab.row(reply)
    same([row["gen_ai_provider_name"], row["input_tokens"], row["agentgateway_group"]], ["openai", 12, "embed"], "embeddings row")


@stub_check
def openai_errors_pass_through(lab):
    error = {"error": {"message": "Rate limit reached", "type": "rate_limit_error", "code": "rate_limit_exceeded", "param": None}}
    lab.stub.force(429, error, {"retry-after": "7"})
    reply = lab.call(fixture("text", "gpt-conformance"), "openai-key")
    upstream(lab, OPENAI)
    same([reply.status, reply.headers.get("retry-after")], [429, "7"], "status and retry-after")
    same(json.loads(reply.raw), error, "error body")


ANTHROPIC_USAGE = {"input_tokens": 8120, "output_tokens": 400, "cached_tokens": 5000}
# The log counts cache reads and writes inside input_tokens for every provider.
ANTHROPIC_TOKENS = {"input_tokens": 8120, "output_tokens": 400, "total_tokens": 8520, "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 3000, "reasoning_tokens": 340}
# 3000 cache-write tokens, 1000 of them at the one-hour rate.
ANTHROPIC_COST = price({"input": 120, "cache_read": 5000, "cache_write": 2000, "cache_write_1h": 1000, "output": 400})


def anthropic_headers(sent, key):
    same(sent["headers"].get("x-api-key"), key, "upstream x-api-key")
    expect("authorization" not in sent["headers"], "the bearer credential must not be forwarded to Anthropic")
    expect(sent["headers"].get("anthropic-version"), "anthropic-version header is missing")


@stub_check
def anthropic_text(lab):
    body = fixture("text", "claude-conformance")
    reply = lab.call(body, "anthropic-key-text")
    sent = upstream(lab, ANTHROPIC)
    anthropic_headers(sent, "anthropic-key-text")
    same(sent["body"], {
        "model": "claude-conformance", "max_tokens": 256, "system": "You are a careful assistant.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with the single word ok."}]}],
    }, "upstream Messages request")
    parsed = reply.parsed()
    same(parsed["text"], stubs.TEXT_MARKER, "reply text")
    same(parsed["usage"], ANTHROPIC_USAGE, "reply usage")
    same([item["type"] for item in parsed["replay"]], ["reasoning", "message"], "reply output items")
    check_row(lab, reply, "anthropic", "claude-conformance", ANTHROPIC_TOKENS, ANTHROPIC_COST)


@stub_check
def anthropic_forced_strict_tool(lab):
    reply = lab.call(fixture("forced-strict-tool", "claude-conformance"), "anthropic-key")
    sent = upstream(lab, ANTHROPIC)
    same(sent["body"]["tools"], [{"name": "probe_tool", "description": "Echo a word back.",
                                  "input_schema": TOOL_SCHEMA, "strict": True}], "upstream tools")
    same(sent["body"]["tool_choice"], {"type": "tool", "name": "probe_tool"}, "upstream tool_choice")
    parsed = reply.parsed()
    same(parsed["calls"], [{"id": "toolu_01conformance", "name": "probe_tool", "arguments": {"echo": stubs.TOOL_MARKER}}], "tool call")
    same([item["type"] for item in parsed["replay"]], ["reasoning", "message", "function_call"], "reply output items")


@stub_check
def anthropic_tool_loop_replays_signed_thinking(lab):
    first_body = fixture("forced-strict-tool", "claude-conformance")
    first = lab.call(first_body, "anthropic-key").parsed()
    upstream(lab, ANTHROPIC)
    second_body = product.replay_input(first_body, first, "Invalid input: echo must be the word pineapple.")
    # Reasoning the provider did not sign (another provider's, here) is dropped, not refused.
    second_body["input"].insert(1, {"type": "reasoning", "id": "rs_other", "summary": [],
                                    "encrypted_content": stubs.OPENAI_REASONING})
    second = lab.call(second_body, "anthropic-key").parsed()
    sent = upstream(lab, ANTHROPIC)
    same(sent["body"]["messages"][1:], [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": stubs.ANTHROPIC_THINKING, "signature": stubs.ANTHROPIC_SIGNATURE},
            {"type": "text", "text": "Checking the supplied evidence."},
            {"type": "tool_use", "id": "toolu_01conformance", "name": "probe_tool", "input": {"echo": stubs.TOOL_MARKER}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01conformance",
                                      "content": "Invalid input: echo must be the word pineapple."}]},
    ], "replayed assistant turn upstream")
    same(second["calls"][0]["arguments"], {"echo": stubs.CORRECTED_MARKER}, "corrected tool call")


@stub_check
def anthropic_effort(lab):
    def thinking(model, effort, max_output_tokens=4096):
        body = fixture("text", model, max_output_tokens=max_output_tokens)
        if effort:
            body["reasoning"] = {"effort": effort}
        lab.call(body, "anthropic-key").parsed()
        sent = upstream(lab, ANTHROPIC)["body"]
        return sent.get("thinking"), sent.get("output_config")

    same(thinking("claude-conformance", "high"), ({"type": "adaptive"}, {"effort": "high"}), "adaptive model, effort high")
    same(thinking("claude-conformance", "minimal"), ({"type": "adaptive"}, {"effort": "low"}), "adaptive model, effort minimal")
    same(thinking("claude-conformance", "none"), (None, None), "adaptive model, effort none")
    same(thinking("claude-conformance", None), (None, None), "adaptive model, no effort")
    same(thinking("claude-conformance-legacy", "medium"), ({"type": "enabled", "budget_tokens": 2048}, None), "budget model, effort medium")


@stub_check
def anthropic_background_request(lab):
    """The product marks background work service_tier=flex. Anthropic has no such tier: it is dropped."""
    reply = lab.call(fixture("background", "claude-conformance"), "anthropic-key")
    sent = upstream(lab, ANTHROPIC)["body"]
    expect("service_tier" not in sent, "service_tier must not be sent to Anthropic")
    same([sent.get("thinking"), sent.get("output_config")], [{"type": "adaptive"}, {"effort": "high"}], "effort upstream")
    same(reply.parsed()["calls"][0]["name"], "probe_tool", "tool call")


@stub_check
def anthropic_text_format(lab):
    body = fixture("text-format", "claude-conformance")
    reply = lab.call(body, "anthropic-key")
    sent = upstream(lab, ANTHROPIC)["body"]
    same(sent.get("output_config"), {"format": {"type": "json_schema", "schema": body["text"]["format"]["schema"]}},
         "upstream output_config")
    same(json.loads(reply.parsed()["text"]), {"echo": stubs.STRUCTURED_MARKER}, "structured reply")


@stub_check
def anthropic_image(lab):
    body = fixture("image", "claude-conformance")
    lab.call(body, "anthropic-key").parsed()
    sent = upstream(lab, ANTHROPIC)["body"]
    data = body["input"][0]["content"][1]["image_url"].split(",", 1)[1]
    same(sent["messages"][0]["content"], [
        {"type": "text", "text": "What colour is this square?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
    ], "upstream image content")


@stub_check
def anthropic_prompt_caching_policy(lab):
    """A route with the promptCaching policy marks the system prompt and the conversation prefix."""
    body = fixture("text", "claude-conformance-cached", instructions="Stay factual and brief. " * 400)
    body["input"] = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": [{"type": "input_text", "text": "Reply with the single word ok."}]},
    ]
    lab.call(body, "anthropic-key").parsed()
    sent = upstream(lab, ANTHROPIC)["body"]
    mark = {"type": "ephemeral"}
    expect(isinstance(sent.get("system"), list) and sent["system"][-1].get("cache_control") == mark,
           "system prompt carries no cache mark: %s" % json.dumps(sent.get("system"))[:200])
    same(sent["messages"][1]["content"][-1].get("cache_control"), mark, "cache mark on the conversation prefix")
    # Without the policy nothing is marked.
    lab.call(dict(body, model="claude-conformance"), "anthropic-key").parsed()
    expect("cache_control" not in json.dumps(upstream(lab, ANTHROPIC)["body"]), "cache marks without the policy")


@stub_check
def anthropic_output_limit(lab):
    lab.stub.force(200, {
        "id": "msg_01limit", "type": "message", "role": "assistant", "model": "claude-conformance",
        "content": [{"type": "text", "text": "The answer begins"}], "stop_reason": "max_tokens",
        "stop_sequence": None, "usage": {"input_tokens": 18, "output_tokens": 4}})
    reply = lab.call(fixture("text", "claude-conformance"), "anthropic-key")
    upstream(lab, ANTHROPIC)
    try:
        product.parse_reply(reply.raw)
    except product.OutputLimit:
        return
    except product.ContractError as err:
        raise Failed("a max_tokens stop must read as the output limit, got: %s" % err)
    raise Failed("a max_tokens stop was reported as a completed reply")


@stub_check
def anthropic_streaming_is_refused(lab):
    reply = lab.call(fixture("text", "claude-conformance", stream=True), "anthropic-key")
    same(lab.stub.take(), [], "upstream requests for a streaming request")
    expect(reply.status == 400 and b"unsupported conversion" in reply.raw,
           "streaming must be refused as an unsupported conversion, got %d %r" % (reply.status, reply.raw[:200]))


@stub_check
def anthropic_errors_keep_the_provider_detail(lab):
    cases = (
        (429, "rate_limit_error", "This request would exceed your organization's rate limit.", "rate_limit_exceeded", {"retry-after": "7"}),
        (401, "authentication_error", "invalid x-api-key", "invalid_api_key", {}),
        (400, "invalid_request_error", "max_tokens: 9999999 > 128000, which is the maximum allowed", "", {}),
        (404, "not_found_error", "model: claude-conformance", "", {}),
        (529, "overloaded_error", "Overloaded", "server_is_overloaded", {}),
    )
    for status, kind, message, code, headers in cases:
        lab.stub.force(status, {"type": "error", "error": {"type": kind, "message": message}}, headers)
        reply = lab.call(fixture("text", "claude-conformance"), "anthropic-key")
        upstream(lab, ANTHROPIC)
        same(reply.status, status, "status for " + kind)
        same(reply.error(), {"code": code, "message": message, "type": kind}, "error for " + kind)
        if headers:
            same(reply.headers.get("retry-after"), headers["retry-after"], "retry-after")


@stub_check
def gemini_text_and_tool(lab):
    reply = lab.call(fixture("text", "gemini-conformance"), "gemini-key-text")
    sent = upstream(lab, GEMINI)
    same(sent["headers"].get("authorization"), "Bearer gemini-key-text", "upstream authorization")
    same(sent["body"]["messages"], [
        {"role": "developer", "content": "You are a careful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Reply with the single word ok."}]},
    ], "upstream messages")
    same(sent["body"].get("max_completion_tokens"), 256, "upstream max_completion_tokens")
    parsed = reply.parsed()
    same(parsed["text"], stubs.TEXT_MARKER, "reply text")
    same(parsed["usage"], {"input_tokens": 8120, "output_tokens": 400, "cached_tokens": 5000}, "reply usage")
    check_row(lab, reply, "gcp.gemini", "gemini-conformance", OPENAI_TOKENS, OPENAI_COST)

    reply = lab.call(fixture("background", "gemini-conformance"), "gemini-key")
    sent = upstream(lab, GEMINI)["body"]
    same(sent["tools"], [{"type": "function", "function": {"name": "probe_tool", "description": "Echo a word back.",
                                                           "parameters": TOOL_SCHEMA, "strict": True}}], "upstream tools")
    same(sent["tool_choice"], {"type": "function", "function": {"name": "probe_tool"}}, "upstream tool_choice")
    same(sent.get("reasoning_effort"), "high", "upstream reasoning_effort")
    same(reply.parsed()["calls"], [{"id": "function-call-conformance", "name": "probe_tool",
                                    "arguments": {"echo": stubs.TOOL_MARKER}}], "tool call")


@stub_check
def gemini_tool_loop_replays_thought_signature(lab):
    first_body = fixture("forced-strict-tool", "gemini-conformance")
    first = lab.call(first_body, "gemini-key").parsed()
    upstream(lab, GEMINI)
    second = lab.call(product.replay_input(first_body, first, "Invalid input: echo must be the word pineapple."), "gemini-key").parsed()
    messages = upstream(lab, GEMINI)["body"]["messages"]
    call = next(m for m in messages if m.get("role") == "assistant")["tool_calls"][0]
    same(call.get("extra_content"), {"google": {"thought_signature": stubs.GEMINI_SIGNATURE}}, "replayed thought signature")
    same(messages[-1], {"role": "tool", "tool_call_id": "function-call-conformance",
                        "content": "Invalid input: echo must be the word pineapple."}, "tool result upstream")
    same(second["calls"][0]["arguments"], {"echo": stubs.CORRECTED_MARKER}, "corrected tool call")


@stub_check
def gemini_text_format(lab):
    body = fixture("text-format", "gemini-conformance")
    reply = lab.call(body, "gemini-key")
    sent = upstream(lab, GEMINI)["body"]
    same((sent.get("response_format") or {}).get("type"), "json_schema", "upstream response_format")
    same(sent["response_format"]["json_schema"].get("schema"), body["text"]["format"]["schema"], "upstream schema")
    same(json.loads(reply.parsed()["text"]), {"echo": stubs.STRUCTURED_MARKER}, "structured reply")


@stub_check
def gemini_errors(lab):
    lab.stub.force(429, [{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).",
                                    "status": "RESOURCE_EXHAUSTED"}}], {"retry-after": "7"})
    reply = lab.call(fixture("text", "gemini-conformance"), "gemini-key")
    upstream(lab, GEMINI)
    same([reply.status, reply.headers.get("retry-after")], [429, "7"], "status and retry-after")
    same(reply.error(), {"code": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted (e.g. check quota).",
                         "type": "rate_limit_error"}, "error")


@stub_check
def gemini_image(lab):
    lab.call(fixture("image", "gemini-conformance"), "gemini-key").parsed()
    content = upstream(lab, GEMINI)["body"]["messages"][-1]["content"]
    if not any(part.get("type") == "image_url" for part in content if isinstance(part, dict)):
        raise Gap("the Responses to chat-completions conversion drops input_image parts, so an image "
                  "never reaches a Gemini model (upstream behaviour, unchanged by the fork)")


@stub_check
def keys_are_not_recorded(lab):
    """The caller's key is a credential for one upstream call. It must not reach the log store or stdout."""
    key = "conformance-key-" + uuid.uuid4().hex
    for model in ("gpt-conformance", "claude-conformance", "gemini-conformance"):
        reply = lab.call(fixture("text", model), key)
        lab.row(reply)
    lab.stub.take()
    stored = lab.sql("SELECT count(*) FROM request_logs WHERE to_jsonb(request_logs)::text LIKE '%%%s%%'" % key)
    same(stored, "0", "request_logs rows containing the caller's key")
    expect(key not in docker("logs", lab.gateway, check=False), "the caller's key appears in the gateway's output")


# ---- live mode ---------------------------------------------------------------------------------

def key_from(variable):
    path = os.environ.get(variable)
    if not path:
        raise RuntimeError("%s must name a file holding the provider key" % variable)
    with open(path) as f:
        key = f.read().strip()
    if not key:
        raise RuntimeError("the file named by %s is empty" % variable)
    SECRETS.append(key)
    return key


def live_row(lab, reply, provider, model, parsed, route="live"):
    row = lab.row(reply)
    same([row["http_status"], row["gen_ai_provider_name"], row["gen_ai_request_model"]], [200, provider, model], "request_logs row")
    same(row["output_tokens"], parsed["usage"]["output_tokens"], "request_logs.output_tokens")
    expect(row["cost"] is not None and row["cost"] > 0, "request_logs.cost is %r" % row["cost"])
    same(row["attributes_json"].get("bondren_route"), route, "request_logs.attributes_json.bondren_route")
    return row


@live_check
def claude_text(lab, keys):
    reply = lab.call(fixture("text", CLAUDE_MODEL), keys["anthropic"], route="live")
    parsed = reply.parsed()
    expect("ok" in parsed["text"].lower(), "unexpected text %r" % parsed["text"])
    live_row(lab, reply, "anthropic", CLAUDE_MODEL, parsed)


@live_check
def claude_forced_strict_tool_as_background(lab, keys):
    parsed = lab.call(fixture("background", CLAUDE_MODEL), keys["anthropic"], route="live").parsed()
    same([(c["name"], c["arguments"]) for c in parsed["calls"]], [("probe_tool", {"echo": "pineapple"})], "tool call")


@live_check
def claude_tool_loop_with_thinking(lab, keys):
    """The provider verifies the signature of replayed thinking, so a 200 on the second step
    means the first step's reasoning survived the trip through the client."""
    first_body = fixture("tool-loop", CLAUDE_MODEL)
    first = lab.call(first_body, keys["anthropic"], route="live").parsed()
    expect(first["calls"], "the model did not call the tool")
    expect(any(item["type"] == "reasoning" and item.get("encrypted_content") for item in first["replay"]),
           "the reply carries no signed reasoning to replay")
    second_body = product.replay_input(first_body, first, "Recorded. Now reply with the single word done.")
    reply = lab.call(second_body, keys["anthropic"], route="live")
    parsed = reply.parsed()
    expect(parsed["text"] or parsed["calls"], "empty second step")
    row = live_row(lab, reply, "anthropic", CLAUDE_MODEL, parsed)
    expect(row["reasoning_tokens"] is not None, "request_logs.reasoning_tokens is missing")


@live_check
def claude_text_format(lab, keys):
    for model in dict.fromkeys((CLAUDE_MODEL, CLAUDE_STRUCTURED_MODEL)):
        parsed = lab.call(fixture("text-format", model), keys["anthropic"], route="live").parsed()
        answer = json.loads(parsed["text"])
        expect(set(answer) == {"city", "population_millions"} and "paris" in answer["city"].lower(),
               "%s: unexpected structured answer %r" % (model, answer))


@live_check
def claude_image(lab, keys):
    parsed = lab.call(fixture("image", CLAUDE_MODEL), keys["anthropic"], route="live").parsed()
    expect("red" in parsed["text"].lower(), "unexpected answer %r" % parsed["text"])


@live_check
def claude_prompt_caching(lab, keys):
    """Second identical request reads the prefix the first one wrote; the row prices both."""
    filler = " ".join("Rule %d: prefer short, factual answers about topic %d." % (n, n * 7919) for n in range(1, 900))
    body = fixture("text", CLAUDE_MODEL, instructions=uuid.uuid4().hex + " " + filler)
    first = lab.call(body, keys["anthropic"], route="live-cached")
    first_usage = json.loads(first.raw).get("usage", {}) if first.status == 200 else {}
    first.parsed()
    expect(first_usage["input_tokens_details"].get("cache_write_tokens", 0) > 0,
           "the first request wrote no cache: %s" % json.dumps(first_usage))
    row = live_row(lab, first, "anthropic", CLAUDE_MODEL, first.parsed(), route="live-cached")
    expect(row["cache_creation_input_tokens"] and row["cache_creation_input_tokens"] > 0, "row has no cache_creation_input_tokens")
    for _ in range(3):
        time.sleep(2)
        second = lab.call(body, keys["anthropic"], route="live-cached")
        parsed = second.parsed()
        if parsed["usage"]["cached_tokens"] > 0:
            row = live_row(lab, second, "anthropic", CLAUDE_MODEL, parsed, route="live-cached")
            same(row["cache_read_input_tokens"], parsed["usage"]["cached_tokens"], "request_logs.cache_read_input_tokens")
            return
    raise Failed("no cache read after three attempts")


@live_check
def claude_errors(lab, keys):
    reply = lab.call(fixture("text", CLAUDE_MODEL), "conformance-invalid-key", route="live")
    same(reply.status, 401, "status for an invalid key")
    error = reply.error()
    expect(error["message"] and error["type"] == "authentication_error" and error["code"] == "invalid_api_key",
           "invalid key error: %s" % json.dumps(error))
    reply = lab.call(fixture("text", "claude-conformance-no-such-model"), keys["anthropic"], route="live")
    same(reply.status, 404, "status for an unknown model")
    expect(reply.error()["type"] == "not_found_error" and reply.error()["message"], "unknown model error: %s" % json.dumps(reply.error()))


@live_check
def gemini_text(lab, keys):
    reply = lab.call(fixture("text", GEMINI_MODEL), keys["gemini"], route="live")
    parsed = reply.parsed()
    expect("ok" in parsed["text"].lower(), "unexpected text %r" % parsed["text"])
    live_row(lab, reply, "gcp.gemini", GEMINI_MODEL, parsed)


@live_check
def gemini_tool_loop(lab, keys):
    first_body = fixture("forced-strict-tool", GEMINI_MODEL)
    first = lab.call(first_body, keys["gemini"], route="live").parsed()
    same([(c["name"], c["arguments"]) for c in first["calls"]], [("probe_tool", {"echo": "pineapple"})], "tool call")
    second_body = product.replay_input(first_body, first, "Recorded.")
    del second_body["tool_choice"]
    second = lab.call(second_body, keys["gemini"], route="live").parsed()
    expect(second["text"] or second["calls"], "empty second step")


@live_check
def gemini_text_format(lab, keys):
    parsed = lab.call(fixture("text-format", GEMINI_MODEL), keys["gemini"], route="live").parsed()
    answer = json.loads(parsed["text"])
    expect("paris" in str(answer.get("city", "")).lower(), "unexpected structured answer %r" % answer)


# ---- runner ------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", help="gateway image reference, ideally pinned by digest")
    parser.add_argument("--live", action="store_true", help="run the live checks instead of the stub checks")
    parser.add_argument("--only", metavar="TEXT", help="run only checks whose name contains TEXT")
    parser.add_argument("--platform", help="docker --platform for the gateway container")
    parser.add_argument("--keep", action="store_true", help="leave the containers running afterwards")
    parser.add_argument("--record", metavar="DIR", help="write every gateway reply to DIR as {status, body} JSON")
    args = parser.parse_args()

    checks = LIVE_CHECKS if args.live else STUB_CHECKS
    checks = [c for c in checks if not args.only or args.only in c.__name__]
    keys = {}
    if args.live:
        if any(c.__name__.startswith("claude") for c in checks):
            keys["anthropic"] = key_from("ANTHROPIC_API_KEY_FILE")
        if any(c.__name__.startswith("gemini") for c in checks):
            keys["gemini"] = key_from("GEMINI_API_KEY_FILE")

    print("conformance: %s mode, image %s" % ("live" if args.live else "stub", args.image))
    passed, failed, gaps = 0, 0, 0
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    with Lab(args.image, args.platform, args.keep, args.record) as lab:
        for check in checks:
            lab.stub.reset()
            lab.check = check.__name__
            try:
                check(lab, keys) if args.live else check(lab)
            except Gap as gap:
                gaps += 1
                print("GAP   %s: %s" % (check.__name__, gap))
            except (Failed, KeyError, IndexError, TypeError, ValueError, StopIteration) as err:
                failed += 1
                print("FAIL  %s: %s" % (check.__name__, scrub("%s: %s" % (type(err).__name__, err))))
            else:
                passed += 1
                print("PASS  %s" % check.__name__)
        if failed:
            print("---- gateway log (tail) ----\n" + lab.logs())
    print("conformance: %d passed, %d failed, %d known gaps" % (passed, failed, gaps))
    return 1 if failed or not passed else 0


if __name__ == "__main__":
    sys.exit(main())
