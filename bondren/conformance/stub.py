#!/usr/bin/env python3
"""Stub provider upstream for the conformance kit.

Answers the provider APIs the gateway calls (OpenAI Responses and embeddings, Anthropic Messages,
Gemini's OpenAI-compatible chat completions) with canned provider-native replies, and keeps every
request it receives in memory so a check can assert what the gateway built.

run.py starts it in-process. `python3 stub.py [PORT]` runs it alone for manual poking.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT_MARKER = "ordinary-text-marker"
TOOL_MARKER = "round-trip-marker"
CORRECTED_MARKER = "corrected-round-trip-marker"
STRUCTURED_MARKER = "structured-output-marker"
OPENAI_REASONING = "encrypted-reasoning-marker"
ANTHROPIC_THINKING = "Weighing the supplied evidence."
ANTHROPIC_SIGNATURE = "CONFORMANCE-SIGNATURE-" + "c2lnbmF0dXJl" * 12
GEMINI_SIGNATURE = "CONFORMANCE-THOUGHT-SIGNATURE-" + "c2lnbmF0dXJl" * 24

# Usage the stub reports. The runner derives the expected request_logs rows from these.
OPENAI_USAGE = {
    "input_tokens": 8120, "output_tokens": 400, "total_tokens": 8520,
    "input_tokens_details": {"cached_tokens": 5000},
    "output_tokens_details": {"reasoning_tokens": 340},
}
ANTHROPIC_USAGE = {
    "input_tokens": 120, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 5000,
    "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 1000},
    "output_tokens": 400, "output_tokens_details": {"thinking_tokens": 340},
    "service_tier": "standard",
}
GEMINI_USAGE = {
    "prompt_tokens": 8120, "completion_tokens": 400, "total_tokens": 8520,
    "prompt_tokens_details": {"cached_tokens": 5000},
    "completion_tokens_details": {"reasoning_tokens": 340},
}


def has_tool_result(req):
    text = json.dumps(req)
    return any(m in text for m in ('"tool_result"', '"function_call_output"', '"role": "tool"'))


def openai_responses(req):
    tools = req.get("tools") or []
    if tools:
        echo = CORRECTED_MARKER if has_tool_result(req) else TOOL_MARKER
        output = [
            {"id": "rs_conformance", "type": "reasoning", "summary": [], "encrypted_content": OPENAI_REASONING},
            {"id": "fc_conformance", "call_id": "call_conformance", "type": "function_call", "status": "completed",
             "name": tools[0].get("name"), "arguments": json.dumps({"echo": echo})},
        ]
    else:
        text = TEXT_MARKER
        if ((req.get("text") or {}).get("format") or {}).get("type") == "json_schema":
            text = json.dumps({"echo": STRUCTURED_MARKER})
        output = [{"id": "msg_conformance", "type": "message", "role": "assistant", "status": "completed",
                   "phase": "final_answer",
                   "content": [{"type": "output_text", "text": text, "annotations": []}]}]
    out = {"id": "resp_conformance", "object": "response", "status": "completed",
           "model": req.get("model", "stub"), "output": output, "usage": OPENAI_USAGE}
    if req.get("service_tier"):
        out["service_tier"] = req["service_tier"]
    return out


def anthropic_messages(req):
    tools = req.get("tools") or []
    name = (req.get("tool_choice") or {}).get("name") or (tools[0].get("name") if tools else None)
    content = [{"type": "thinking", "thinking": ANTHROPIC_THINKING, "signature": ANTHROPIC_SIGNATURE}]
    stop = "end_turn"
    if name:
        echo = CORRECTED_MARKER if has_tool_result(req) else TOOL_MARKER
        content.append({"type": "text", "text": "Checking the supplied evidence."})
        content.append({"type": "tool_use", "id": "toolu_01conformance", "name": name, "input": {"echo": echo}})
        stop = "tool_use"
    elif ((req.get("output_config") or {}).get("format") or {}).get("type") == "json_schema":
        content.append({"type": "text", "text": json.dumps({"echo": STRUCTURED_MARKER})})
    else:
        content.append({"type": "text", "text": TEXT_MARKER})
    return {"id": "msg_01conformance", "type": "message", "role": "assistant",
            "model": req.get("model", "stub"), "content": content, "stop_reason": stop,
            "stop_sequence": None, "usage": ANTHROPIC_USAGE}


def gemini_chat_completions(req):
    tools = req.get("tools") or []
    message = {"role": "assistant", "content": None}
    finish = "stop"
    if tools:
        echo = CORRECTED_MARKER if has_tool_result(req) else TOOL_MARKER
        function = tools[0].get("function") or tools[0]
        message["tool_calls"] = [{
            "id": "function-call-conformance", "type": "function",
            "function": {"name": function.get("name"), "arguments": json.dumps({"echo": echo})},
            "extra_content": {"google": {"thought_signature": GEMINI_SIGNATURE}},
        }]
        finish = "tool_calls"
    elif (req.get("response_format") or {}).get("type") == "json_schema":
        message["content"] = json.dumps({"echo": STRUCTURED_MARKER})
    else:
        message["content"] = TEXT_MARKER
    return {"id": "chatcmpl-conformance", "object": "chat.completion", "created": 1,
            "model": req.get("model", "stub"),
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": GEMINI_USAGE}


def embeddings(req):
    inputs = req.get("input") or []
    if isinstance(inputs, str):
        inputs = [inputs]
    vector = [0.0] * int(req.get("dimensions") or 8)
    vector[0] = vector[-1] = 0.25
    return {"object": "list", "model": req.get("model", "stub"),
            "data": [{"object": "embedding", "index": i, "embedding": vector} for i in range(len(inputs))],
            "usage": {"prompt_tokens": 12, "total_tokens": 12}}


ROUTES = {
    "/openai/v1/responses": openai_responses,
    "/openai/v1/embeddings": embeddings,
    "/anthropic/v1/messages": anthropic_messages,
    "/gemini/v1beta/openai/chat/completions": gemini_chat_completions,
}


class Stub:
    """An HTTP server on a background thread. `requests` holds what arrived since `take()`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests = []
        self._forced = None
        self._server = None

    def start(self, host="127.0.0.1", port=0):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                path = self.path.split("?")[0]
                with stub._lock:
                    stub._requests.append({
                        "path": path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    })
                    forced, stub._forced = stub._forced, None
                if forced:
                    status, headers, reply = forced
                elif path in ROUTES and isinstance(body, dict):
                    status, headers, reply = 200, {}, ROUTES[path](body)
                else:
                    status, headers, reply = 404, {}, {"error": {"message": "conformance stub: no answer for " + path}}
                payload = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self._server.server_address[1]

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def take(self):
        """Requests received since the last call."""
        with self._lock:
            requests, self._requests = self._requests, []
        return requests

    def reset(self):
        """Forget recorded requests and any pending forced reply."""
        with self._lock:
            self._requests, self._forced = [], None

    def force(self, status, reply, headers=None):
        """Answer the next request with this status and JSON body instead of the canned reply."""
        with self._lock:
            self._forced = (status, headers or {}, reply)


if __name__ == "__main__":
    stub = Stub()
    print("conformance stub on port", stub.start(port=int(sys.argv[1]) if len(sys.argv) > 1 else 9000))
    threading.Event().wait()
