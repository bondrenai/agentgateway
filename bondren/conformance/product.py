"""The product's side of the wire.

A Python mirror of how the Bondren API reads a gateway reply (`internal/inferencewire/wire.go`:
parseCompletionResponse, projectOutput, providerErrorDetail). It is deliberately as strict as the
product: a reply this module rejects is a reply the product would treat as a failed call. Keep it
in step with that file; the product's own contract test stays the authority.
"""
import json


class ContractError(Exception):
    """The product would reject this reply."""


class OutputLimit(ContractError):
    """The reply stopped at max_output_tokens: the product keeps the usage and drops the output."""


def _object(raw):
    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ContractError("duplicate key in JSON object")
        return dict(pairs)

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except ValueError as err:
        raise ContractError("reply is not JSON: %s" % err)
    if not isinstance(value, dict):
        raise ContractError("reply is not a JSON object")
    return value


def _only_keys(item, allowed, what):
    extra = sorted(set(item) - set(allowed))
    if extra:
        raise ContractError("%s carries keys the product rejects: %s" % (what, extra))


def _count(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("%s is not a token count: %r" % (what, value))
    return value


def _usage(usage):
    if not isinstance(usage, dict):
        raise ContractError("usage is not an object")
    out = {
        "input_tokens": _count(usage.get("input_tokens"), "usage.input_tokens"),
        "output_tokens": _count(usage.get("output_tokens"), "usage.output_tokens"),
        "cached_tokens": 0,
    }
    details = usage.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise ContractError("usage.input_tokens_details is not an object")
        if "cached_tokens" in details:
            out["cached_tokens"] = _count(details["cached_tokens"], "cached_tokens")
            if out["cached_tokens"] > out["input_tokens"]:
                raise ContractError("cached_tokens exceeds input_tokens")
    return out


def _reasoning(item):
    _only_keys(item, ("type", "id", "summary", "content", "encrypted_content", "status"), "reasoning item")
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        raise ContractError("reasoning item has no id")
    if item.get("status") not in (None, "completed"):
        raise ContractError("reasoning item status is %r" % item.get("status"))
    if not isinstance(item.get("encrypted_content"), (str, type(None))):
        raise ContractError("reasoning encrypted_content is not a string")
    for field, kind in (("summary", "summary_text"), ("content", "reasoning_text")):
        parts = item.get(field)
        if field == "content" and parts is None:
            continue
        if not isinstance(parts, list):
            raise ContractError("reasoning %s is not a list" % field)
        for part in parts:
            if not isinstance(part, dict):
                raise ContractError("reasoning %s part is not an object" % field)
            _only_keys(part, ("type", "text"), "reasoning %s part" % field)
            if part.get("type") != kind or not isinstance(part.get("text"), str):
                raise ContractError("reasoning %s part is not %s" % (field, kind))


def _message(item):
    _only_keys(item, ("type", "id", "role", "status", "phase", "content"), "message item")
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        raise ContractError("message item has no id")
    if item.get("role") != "assistant" or item.get("status") != "completed":
        raise ContractError("message item is not a completed assistant message")
    if item.get("phase") not in (None, "commentary", "final_answer"):
        raise ContractError("message phase is %r" % item.get("phase"))
    content = item.get("content")
    if not isinstance(content, list) or not content:
        raise ContractError("message item has no content")
    text = ""
    for part in content:
        if not isinstance(part, dict):
            raise ContractError("message content part is not an object")
        _only_keys(part, ("type", "text", "annotations", "logprobs"), "message content part")
        if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
            raise ContractError("message content part is not output_text")
        for field in ("annotations", "logprobs"):
            if not isinstance(part.get(field), (list, type(None))):
                raise ContractError("message content %s is not a list" % field)
        text += part["text"]
    return text


def _function_call(item, seen):
    _only_keys(item, ("type", "id", "call_id", "name", "arguments", "status"), "function_call item")
    if item.get("status") not in (None, "completed"):
        raise ContractError("function_call status is %r" % item.get("status"))
    for field in ("call_id", "name", "arguments"):
        if not isinstance(item.get(field), str):
            raise ContractError("function_call %s is not a string" % field)
    if not isinstance(item.get("id"), (str, type(None))):
        raise ContractError("function_call id is not a string")
    if not item["call_id"].strip() or not item["name"].strip() or item["call_id"] in seen:
        raise ContractError("function_call has a blank or repeated call_id or name")
    try:
        arguments = json.loads(item["arguments"])
    except ValueError:
        arguments = None
    if not isinstance(arguments, dict):
        raise ContractError("function_call arguments are not a JSON object")
    seen.add(item["call_id"])
    return {"id": item["call_id"], "name": item["name"], "arguments": arguments}


def project_output(output):
    """Text and tool calls of an output list; raises when an item is not one the product accepts."""
    text, calls, seen = "", [], set()
    for item in output:
        if not isinstance(item, dict):
            raise ContractError("output item is not an object")
        kind = item.get("type")
        if kind == "reasoning":
            _reasoning(item)
        elif kind == "message":
            text += _message(item)
        elif kind == "function_call":
            calls.append(_function_call(item, seen))
        else:
            raise ContractError("output item type %r is not accepted" % kind)
    if not text and not calls:
        raise ContractError("reply has neither text nor tool calls")
    return text, calls


def parse_reply(raw):
    """A 2xx gateway reply as the product reads it: text, tool calls, usage and the replay items."""
    reply = _object(raw)
    usage = _usage(reply["usage"]) if reply.get("usage") is not None else None
    status = reply.get("status")
    if not isinstance(status, str):
        raise ContractError("reply has no status")
    details = reply.get("incomplete_details")
    if status == "incomplete" and reply.get("error") is None:
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            raise OutputLimit("reply stopped at max_output_tokens")
    if status != "completed" or reply.get("error") is not None or details is not None:
        raise ContractError("reply status is %r" % status)
    if not isinstance(reply.get("id"), str) or not reply["id"].strip():
        raise ContractError("reply has no id")
    output = reply.get("output")
    if not isinstance(output, list):
        raise ContractError("reply output is not a list")
    text, calls = project_output(output)
    return {"id": reply["id"], "text": text, "calls": calls, "usage": usage, "replay": output}


def error_detail(raw):
    """The bounded OpenAI-style error fields the product logs and acts on from a non-2xx body."""
    try:
        error = json.loads(raw).get("error")
    except (ValueError, AttributeError):
        return {"code": "", "message": "", "type": ""}
    if not isinstance(error, dict):
        return {"code": "", "message": "", "type": ""}
    return {field: error[field] if isinstance(error.get(field), str) else ""
            for field in ("code", "message", "type")}


def replay_input(first_request, reply, tool_output):
    """The follow-up request of a tool loop: the reply's output items go back verbatim, then one
    function_call_output per call (the product replays, it does not rebuild)."""
    follow_up = json.loads(json.dumps(first_request))
    follow_up["input"] = list(first_request["input"]) + list(reply["replay"]) + [
        {"type": "function_call_output", "call_id": call["id"], "output": tool_output}
        for call in reply["calls"]
    ]
    return follow_up
