from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Generator as TypingGenerator, Iterable, Optional, TypedDict
import inspect
from typing import get_args, get_origin, get_type_hints

ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


Template = Any


class ToolFunctionSpec(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolSpec(TypedDict, total=False):
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    returns: str


ToolLike = ToolSpec | Callable[..., Any]


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e


def _http_post_sse(
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout_s: float = 300.0,
) -> TypingGenerator[dict[str, Any], None, None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e


def _http_post_ndjson(
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout_s: float = 300.0,
) -> TypingGenerator[dict[str, Any], None, None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e


def _pytype_to_json_schema(t: Any) -> dict[str, Any]:
    if t is Any or t is inspect._empty:
        return {
            "type": ["string", "number", "integer", "boolean", "object", "array", "null"],
        }

    origin = get_origin(t)
    args = get_args(t)

    # Optional / Union
    if origin is None and isinstance(t, type):
        if t is str:
            return {"type": "string"}
        if t is int:
            return {"type": "integer"}
        if t is float:
            return {"type": "number"}
        if t is bool:
            return {"type": "boolean"}
        if t is dict:
            return {"type": "object", "additionalProperties": True}
        if t is list:
            return {"type": "array", "items": {}}
        if issubclass(t, TypedDict):
            hints = get_type_hints(t, include_extras=True)
            required_keys = getattr(t, "__required_keys__", set(hints.keys()))
            props = {k: _pytype_to_json_schema(v) for k, v in hints.items()}
            return {
                "type": "object",
                "properties": props,
                "required": sorted(list(required_keys)),
                "additionalProperties": False,
            }
        return {"type": "object"}

    if origin is list or origin is list.__class__ or origin is Iterable:
        item_t = args[0] if args else Any
        return {"type": "array", "items": _pytype_to_json_schema(item_t)}

    if origin is dict:
        val_t = args[1] if len(args) >= 2 else Any
        return {"type": "object", "additionalProperties": _pytype_to_json_schema(val_t)}

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _pytype_to_json_schema(args[0])}
        return {"type": "array", "prefixItems": [_pytype_to_json_schema(a) for a in args]}

    if str(origin).endswith("Literal"):
        return {"enum": list(args)}

    if origin is getattr(__import__("typing"), "Union", None) or origin is type(Optional[int]):  # defensive
        schemas = []
        for a in args:
            if a is type(None):
                schemas.append({"type": "null"})
            else:
                schemas.append(_pytype_to_json_schema(a))
        return {"anyOf": schemas} if schemas else {"type": "null"}

    return {"type": ["string", "number", "integer", "boolean", "object", "array", "null"]}


def _is_optional_annotation(t: Any) -> bool:
    origin = get_origin(t)
    if origin is getattr(__import__("typing"), "Union", None):
        return any(a is type(None) for a in get_args(t))
    return False


def _function_to_toolspec(func: Callable[..., Any]) -> ToolSpec:
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    doc = inspect.getdoc(func) or ""
    name = getattr(func, "__name__", "tool")

    properties: dict[str, Any] = {}
    required: list[str] = []
    additional_properties = False

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                additional_properties = True
            continue

        ann = hints.get(p.name, p.annotation)
        properties[p.name] = _pytype_to_json_schema(ann)

        has_default = p.default is not inspect._empty
        if not has_default and not _is_optional_annotation(ann):
            required.append(p.name)

    params_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True if additional_properties else False,
    }

    ret_ann = hints.get("return", sig.return_annotation)
    returns_str = "" if ret_ann in (inspect._empty, Any) else getattr(ret_ann, "__name__", str(ret_ann))

    description = doc.strip()
    if returns_str and returns_str not in description:
        description = (description + f"\n\nReturns: {returns_str}").strip()

    return {
        "name": str(name),
        "description": description,
        "parameters": params_schema,
        "func": func,
        "returns": returns_str,
    }


def _normalize_tools(tools: Optional[Iterable[ToolLike]]) -> list[ToolSpec]:
    if not tools:
        return []
    normalized: list[ToolSpec] = []
    for t in tools:
        if callable(t) and not isinstance(t, dict):
            normalized.append(_function_to_toolspec(t))
            continue

        if not isinstance(t, dict):
            raise TypeError("tools must be an iterable of callables or dict ToolSpec")

        if "func" not in t or not callable(t["func"]):
            raise ValueError(f"tool '{t.get('name')}' is missing a callable 'func'")

        if "name" not in t or not t["name"]:
            inferred = getattr(t["func"], "__name__", "tool")
            t = dict(t)
            t["name"] = str(inferred)

        normalized.append(
            {
                "name": str(t["name"]),
                "description": str(t.get("description", "") or (inspect.getdoc(t["func"]) or "")),
                "parameters": t.get("parameters")
                or _function_to_toolspec(t["func"]).get("parameters")  # infer if missing
                or {"type": "object", "properties": {}, "additionalProperties": True},
                "func": t["func"],
                "returns": str(t.get("returns", "")),
            }
        )
    return normalized


def _openai_tools_payload(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters")
                or {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }
        for t in tools
    ]


def _tool_map(tools: list[ToolSpec]) -> dict[str, Callable[..., Any]]:
    return {t["name"]: t["func"] for t in tools}


def _call_tool(func: Callable[..., Any], args: Any) -> Any:
    if args is None:
        return func()
    if isinstance(args, dict):
        try:
            return func(**args)
        except TypeError:
            return func(args)
    return func(args)


def _try_parse_json_obj(text: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        parsed = _try_parse_json_obj(candidate)
        if parsed is not None:
            return parsed

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                parsed = _try_parse_json_obj(candidate)
                if parsed is not None:
                    return parsed
                return None
    return None


def _template_to_json_schema(template: Template) -> dict[str, Any]:
    def make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
        # JSON Schema: allow null by using type as a list (draft-07 style).
        t = schema.get("type")
        if isinstance(t, str):
            schema = dict(schema)
            schema["type"] = [t, "null"]
            return schema
        if isinstance(t, list):
            if "null" not in t:
                schema = dict(schema)
                schema["type"] = [*t, "null"]
            return schema
        return {"anyOf": [schema, {"type": "null"}]}

    def schema_of(t: Template) -> dict[str, Any]:
        if t is str:
            return make_nullable({"type": "string"})
        if t is int:
            return make_nullable({"type": "integer"})
        if t is float:
            return make_nullable({"type": "number"})
        if t is bool:
            return make_nullable({"type": "boolean"})
        if isinstance(t, dict):
            props = {str(k): schema_of(v) for k, v in t.items()}
            return make_nullable(
                {
                "type": "object",
                "properties": props,
                "required": list(props.keys()),
                "additionalProperties": False,
                }
            )
        if isinstance(t, list):
            if len(t) != 1:
                raise TypeError("List template must have exactly one element, e.g. [str]")
            return make_nullable({"type": "array", "items": schema_of(t[0])})
        raise TypeError(
            "Unsupported template. Use primitives (str/int/float/bool), dict, or single-item list."
        )

    schema = schema_of(template)
    if schema.get("type") != "object" and schema.get("type") != ["object", "null"]:
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
        }
    return schema

 
def _coerce_value(value: Any, template: Template) -> Any:
    if value is None:
        return None
    if template is str:
        return "" if value is None else str(value)
    if template is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            m = re.search(r"-?\d+", value)
            if not m:
                raise ValueError(f"Cannot coerce to int: {value!r}")
            return int(m.group(0))
        raise ValueError(f"Cannot coerce to int: {value!r}")
    if template is float:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            m = re.search(r"-?\d+(?:\.\d+)?", value)
            if not m:
                raise ValueError(f"Cannot coerce to float: {value!r}")
            return float(m.group(0))
        raise ValueError(f"Cannot coerce to float: {value!r}")
    if template is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "yes", "y", "1", "vrai", "oui"}:
                return True
            if v in {"false", "no", "n", "0", "faux", "non"}:
                return False
        raise ValueError(f"Cannot coerce to bool: {value!r}")
    if isinstance(template, dict):
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        out: dict[str, Any] = {}
        for k, sub_t in template.items():
            out[str(k)] = _coerce_value(value.get(k), sub_t)
        return out
    if isinstance(template, list):
        if len(template) != 1:
            raise TypeError("List template must have exactly one element")
        item_t = template[0]
        if not isinstance(value, list):
            raise ValueError("Expected array")
        return [_coerce_value(v, item_t) for v in value]
    raise TypeError("Unsupported template")


def _coerce_and_validate_object(obj: dict[str, Any], template: Template) -> dict[str, Any]:
    schema = _template_to_json_schema(template)
    props = schema.get("properties") or {}
    if set(props.keys()) == {"value"} and not isinstance(template, dict):
        coerced = _coerce_value(obj.get("value"), template)
        return {"value": coerced}
    if not isinstance(template, dict):
        return _coerce_value(obj, template)  # type: ignore[return-value]
    return _coerce_value(obj, template)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class Generator(ABC):
    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt
        self.model_name = model_name

    def _prepend_system(self, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.system_prompt:
            return list(conversation)
        if conversation and conversation[0].get("role") == ROLE_SYSTEM:
            return list(conversation)
        return [{"role": ROLE_SYSTEM, "content": self.system_prompt}] + list(conversation)

    def answer(
        self,
        prompt: str,
        tools: Optional[Iterable[ToolLike]] = None,
        *,
        stream: bool = False,
    ) -> str | TypingGenerator[str, None, None]:
        messages = self._prepend_system([{"role": ROLE_USER, "content": prompt}])
        return self.chat(messages, tools=tools, stream=stream)

    def stream_answer(
        self, prompt: str, tools: Optional[Iterable[ToolLike]] = None
    ) -> TypingGenerator[str, None, None]:
        out = self.answer(prompt, tools=tools, stream=True)
        if isinstance(out, str):
            yield out
        else:
            yield from out

    def chat(
        self,
        conversation: list[dict[str, Any]],
        tools: Optional[Iterable[ToolLike]] = None,
        *,
        stream: bool = False,
    ) -> str | TypingGenerator[str, None, None]:
        if stream:
            return self._stream_chat(conversation, tools=tools)
        return self._chat(conversation, tools=tools)

    def stream_chat(
        self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None
    ) -> TypingGenerator[str, None, None]:
        out = self.chat(conversation, tools=tools, stream=True)
        if isinstance(out, str):
            yield out
        else:
            yield from out

    @abstractmethod
    def _chat(self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def _stream_chat(
        self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None
    ) -> TypingGenerator[str, None, None]:
        raise NotImplementedError

    @abstractmethod
    def parse(
        self, text: str, format: dict[str, Any], tools: Optional[Iterable[ToolLike]] = None
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIGenerator(Generator):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        max_tool_rounds: int = 8,
    ):
        super().__init__(model_name=model_name, system_prompt=system_prompt)
        if not api_key:
            raise ValueError("OpenAI api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tool_rounds = max_tool_rounds

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        *,
        response_format: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if tools:
            payload["tools"] = _openai_tools_payload(tools)
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.base_url}/chat/completions"
        return _http_post_json(url, payload, headers=self._headers(), timeout_s=self.timeout_s)

    def _extract_openai_message(self, resp: dict[str, Any]) -> dict[str, Any]:
        try:
            return resp["choices"][0]["message"]
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenAI response: {resp}") from e

    def _openai_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls = message.get("tool_calls") or []
        parsed: list[ToolCall] = []
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name") or ""
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}
            parsed.append(ToolCall(id=str(tc.get("id") or ""), name=str(name), arguments=args))
        return parsed

    def _chat(self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None) -> str:
        tools_list = _normalize_tools(tools)
        tool_funcs = _tool_map(tools_list)

        messages = self._prepend_system(conversation)
        for _round in range(self.max_tool_rounds + 1):
            resp = self._chat_once(messages, tools_list)
            msg = self._extract_openai_message(resp)

            tool_calls = self._openai_tool_calls_from_message(msg)
            if not tool_calls:
                return (msg.get("content") or "").strip()

            messages.append({"role": ROLE_ASSISTANT, "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})

            for tc in tool_calls:
                func = tool_funcs.get(tc.name)
                if not func:
                    tool_out = {"error": f"Tool not found: {tc.name}"}
                else:
                    try:
                        tool_out = _call_tool(func, tc.arguments)
                    except Exception as e:
                        tool_out = {"error": str(e)}

                messages.append(
                    {
                        "role": ROLE_TOOL,
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_out, ensure_ascii=False)
                        if not isinstance(tool_out, str)
                        else tool_out,
                    }
                )

        raise RuntimeError("Too many tool-calling rounds")

    def _stream_chat(
        self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None
    ) -> TypingGenerator[str, None, None]:
        tools_list = _normalize_tools(tools)
        tool_funcs = _tool_map(tools_list)

        messages = self._prepend_system(conversation)
        rounds = 0
        while rounds <= self.max_tool_rounds:
            rounds += 1

            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
            }
            if tools_list:
                payload["tools"] = _openai_tools_payload(tools_list)
                payload["tool_choice"] = "auto"

            url = f"{self.base_url}/chat/completions"

            tool_calls_acc: dict[int, dict[str, Any]] = {}
            assistant_content_parts: list[str] = []

            for event in _http_post_sse(url, payload, headers=self._headers(), timeout_s=self.timeout_s * 3):
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunk = delta["content"]
                    assistant_content_parts.append(chunk)
                    yield chunk

                # Tool call deltas can arrive chunked
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    entry = tool_calls_acc.setdefault(idx, {"id": "", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]

            tool_calls_sorted = [tool_calls_acc[k] for k in sorted(tool_calls_acc.keys())]
            if not tool_calls_sorted:
                return

            messages.append(
                {
                    "role": ROLE_ASSISTANT,
                    "content": "".join(assistant_content_parts) if assistant_content_parts else None,
                    "tool_calls": tool_calls_sorted,
                }
            )

            for tc_raw in tool_calls_sorted:
                tc_id = str(tc_raw.get("id") or "")
                fn = tc_raw.get("function") or {}
                name = str(fn.get("name") or "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}

                func = tool_funcs.get(name)
                if not func:
                    tool_out = {"error": f"Tool not found: {name}"}
                else:
                    try:
                        tool_out = _call_tool(func, args)
                    except Exception as e:
                        tool_out = {"error": str(e)}

                messages.append(
                    {
                        "role": ROLE_TOOL,
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_out, ensure_ascii=False)
                        if not isinstance(tool_out, str)
                        else tool_out,
                    }
                )

        raise RuntimeError("Too many tool-calling rounds (stream)")

    def parse(
        self, text: str, format: dict[str, Any], tools: Optional[Iterable[ToolLike]] = None
    ) -> dict[str, Any]:
        schema = _template_to_json_schema(format)

        # 1) Try direct parse
        if parsed := _try_parse_json_obj(text):
            try:
                return _coerce_and_validate_object(parsed, format)
            except Exception:
                pass

        # 2) Try extract JSON from text
        if extracted := _extract_first_json_object(text):
            try:
                return _coerce_and_validate_object(extracted, format)
            except Exception:
                pass

        # 3) Ask the model to extract fields matching the template
        tools_list = _normalize_tools(tools)
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        template_str = json.dumps(
            {k: getattr(v, "__name__", str(v)) for k, v in (format or {}).items()},
            ensure_ascii=False,
            indent=2,
        )

        messages: list[dict[str, Any]] = self._prepend_system(
            [
                {
                    "role": ROLE_USER,
                    "content": (
                        "Extract structured data from the TEXT according to the TEMPLATE. "
                        "Return ONLY a valid JSON object matching the TEMPLATE and SCHEMA.\n\n"
                        f"TEMPLATE (key -> type):\n{template_str}\n\n"
                        f"SCHEMA (JSON Schema):\n{schema_str}\n\n"
                        f"TEXT:\n{text}"
                    ),
                }
            ]
        )

        resp = self._chat_once(messages, tools_list, response_format={"type": "json_object"})
        msg = self._extract_openai_message(resp)
        content = (msg.get("content") or "").strip()
        parsed = _try_parse_json_obj(content) or _extract_first_json_object(content)
        if parsed is None:
            raise ValueError("Model did not return valid JSON object")
        return _coerce_and_validate_object(parsed, format)


class OllamaGenerator(Generator):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        max_tool_rounds: int = 8,
        keep_alive: str = "5m",
    ):
        super().__init__(model_name=model_name, system_prompt=system_prompt)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tool_rounds = max_tool_rounds
        self.keep_alive = keep_alive

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        *,
        stream: bool = False,
        format: Optional[Any] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = _openai_tools_payload(tools)
        if format is not None:
            payload["format"] = format

        url = f"{self.base_url}/api/chat"
        if stream:
            raise RuntimeError("_chat_once does not support stream=True")
        return _http_post_json(url, payload, timeout_s=self.timeout_s)

    def _ollama_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        *,
        format: Optional[Any] = None,
    ) -> TypingGenerator[dict[str, Any], None, None]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = _openai_tools_payload(tools)
        if format is not None:
            payload["format"] = format
        url = f"{self.base_url}/api/chat"
        yield from _http_post_ndjson(url, payload, timeout_s=self.timeout_s * 3)

    def _tool_calls_from_ollama(self, resp: dict[str, Any]) -> list[ToolCall]:
        msg = resp.get("message") or {}
        tcs = msg.get("tool_calls") or []
        out: list[ToolCall] = []
        for i, tc in enumerate(tcs):
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            out.append(ToolCall(id=str(tc.get("id") or f"toolcall_{i}"), name=name, arguments=args))
        return out

    def _chat(self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None) -> str:
        tools_list = _normalize_tools(tools)
        tool_funcs = _tool_map(tools_list)

        messages = self._prepend_system(conversation)
        for _round in range(self.max_tool_rounds + 1):
            resp = self._chat_once(messages, tools_list)
            msg = (resp.get("message") or {})
            tool_calls = self._tool_calls_from_ollama(resp)

            if not tool_calls:
                return str(msg.get("content") or "").strip()

            messages.append({"role": ROLE_ASSISTANT, "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")})

            for tc in tool_calls:
                func = tool_funcs.get(tc.name)
                if not func:
                    tool_out = {"error": f"Tool not found: {tc.name}"}
                else:
                    try:
                        tool_out = _call_tool(func, tc.arguments)
                    except Exception as e:
                        tool_out = {"error": str(e)}
                messages.append(
                    {
                        "role": ROLE_TOOL,
                        "content": json.dumps(tool_out, ensure_ascii=False)
                        if not isinstance(tool_out, str)
                        else tool_out,
                    }
                )

        raise RuntimeError("Too many tool-calling rounds")

    def _stream_chat(
        self, conversation: list[dict[str, Any]], tools: Optional[Iterable[ToolLike]] = None
    ) -> TypingGenerator[str, None, None]:
        tools_list = _normalize_tools(tools)
        tool_funcs = _tool_map(tools_list)

        messages = self._prepend_system(conversation)
        rounds = 0
        while rounds <= self.max_tool_rounds:
            rounds += 1
            assistant_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for event in self._ollama_stream(messages, tools_list):
                msg = event.get("message") or {}
                if msg.get("content"):
                    chunk = str(msg["content"])
                    assistant_chunks.append(chunk)
                    yield chunk
                if msg.get("tool_calls"):
                    tool_calls = msg.get("tool_calls") or []
                if event.get("done") is True:
                    break

            if not tool_calls:
                return

            messages.append({"role": ROLE_ASSISTANT, "content": "".join(assistant_chunks), "tool_calls": tool_calls})

            # Execute tools
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                args = fn.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}

                func = tool_funcs.get(name)
                if not func:
                    tool_out = {"error": f"Tool not found: {name}"}
                else:
                    try:
                        tool_out = _call_tool(func, args)
                    except Exception as e:
                        tool_out = {"error": str(e)}

                messages.append(
                    {
                        "role": ROLE_TOOL,
                        "content": json.dumps(tool_out, ensure_ascii=False)
                        if not isinstance(tool_out, str)
                        else tool_out,
                    }
                )

            time.sleep(0.01)

        raise RuntimeError("Too many tool-calling rounds (stream)")

    def parse(
        self, text: str, format: dict[str, Any], tools: Optional[Iterable[ToolLike]] = None
    ) -> dict[str, Any]:
        schema = _template_to_json_schema(format)

        if parsed := _try_parse_json_obj(text):
            try:
                return _coerce_and_validate_object(parsed, format)
            except Exception:
                pass
        if extracted := _extract_first_json_object(text):
            try:
                return _coerce_and_validate_object(extracted, format)
            except Exception:
                pass

        tools_list = _normalize_tools(tools)

        template_str = json.dumps(
            {k: getattr(v, "__name__", str(v)) for k, v in (format or {}).items()},
            ensure_ascii=False,
            indent=2,
        )
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        messages: list[dict[str, Any]] = self._prepend_system(
            [
                {
                    "role": ROLE_USER,
                    "content": (
                        "Extract structured data from the TEXT according to the TEMPLATE. "
                        "Return ONLY a valid JSON object matching the TEMPLATE and SCHEMA.\n\n"
                        f"TEMPLATE (key -> type):\n{template_str}\n\n"
                        f"SCHEMA (JSON Schema):\n{schema_str}\n\n"
                        f"TEXT:\n{text}"
                    ),
                }
            ]
        )

        resp = self._chat_once(messages, tools_list, format=schema)
        content = str((resp.get("message") or {}).get("content") or "").strip()
        parsed = _try_parse_json_obj(content) or _extract_first_json_object(content)
        if parsed is None:
            raise ValueError("Model did not return valid JSON object")
        return _coerce_and_validate_object(parsed, format)
