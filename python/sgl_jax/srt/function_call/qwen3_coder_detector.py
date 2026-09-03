"""Detector for the Qwen3-Coder / Qwen3.8 native tool-call format.

    <tool_call>
    <function=name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

Streaming uses a cursor-based state machine (backported from upstream
sglang): ``parsed_pos`` walks the buffer tag-by-tag; a parameter block ends
at ``</parameter>``, the next ``<parameter=``, ``</function>``, or
end-of-text; values are converted according to the tool schema, so JSON
values of arbitrary nesting (objects/arrays/strings) are decoded once at
block end -- no per-token reparsing of accumulated JSON.
"""

import html
import json
import logging
import os
import re
from typing import Any

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sgl_jax.srt.function_call.ebnf_composer import EBNFComposer
from sgl_jax.srt.function_call.utils import (
    get_schema_properties,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)

# Instrumentation: set SGLANG_TOOL_PARSE_DEBUG=1 to trace every delta and
# parser state transition (used to diagnose argument-loss reports).
_TOOL_PARSE_DEBUG = os.environ.get("SGLANG_TOOL_PARSE_DEBUG", "") == "1"


def _dbg(msg: str) -> None:
    if _TOOL_PARSE_DEBUG:
        logger.info("[tool-parse] %s", msg)


def _safe_val(raw: str) -> Any:
    raw = html.unescape(raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return safe_literal_eval(raw)
    except Exception:
        return raw


# Parameter blocks terminate at </parameter>, the next <parameter=,
# </function>, or end-of-text (backported from upstream sglang).
_PARAM_START_RE = re.compile(r"<parameter=([^>]+)>")


def _trim_leaf_text(text: str) -> str:
    """Trim exactly one leading/trailing newline from a leaf parameter value
    (the model emits values on their own lines)."""
    if text.startswith(chr(10)):
        text = text[1:]
    if text.endswith(chr(10)):
        text = text[:-1]
    return text


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        # Parameter blocks terminate at </parameter>, the next <parameter=,
        # </function>, or end-of-text (backported from upstream sglang).
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )
        self._buf: str = ""

        # Cursor-based streaming state (backported from upstream sglang):
        # parsed_pos walks self._buffer tag-by-tag. Parameter values are
        # converted according to the tool schema when their region completes,
        # so JSON values of arbitrary nesting (objects/arrays/strings) are
        # decoded once -- no per-token reparsing of accumulated JSON.
        self.parsed_pos: int = 0
        self.is_inside_tool_call: bool = False
        self.json_started: bool = False
        self.current_tool_param_count: int = 0
        self.current_func_name: str | None = None
        self.current_tool_args_closed: bool = True
        # Open parameter bookkeeping (for flush_pending at generation end).
        self._open_param_name: str | None = None
        self._open_param_value_start: int | None = None
        # Text discarded between <function=...> and </function> when the
        # model omits <parameter> wrappers entirely (bare JSON body).
        self._pending_body_text: str = ""

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    # ------------------------------------------------------------------
    # Schema-driven value conversion (backported from upstream sglang)
    # ------------------------------------------------------------------
    def _get_arguments_config(self, func_name: str | None) -> dict:
        if not func_name:
            return {}
        for tool in getattr(self, "_tools", None) or []:
            if tool.function.name == func_name:
                params = tool.function.parameters
                if isinstance(params, dict):
                    # Descend into anyOf/oneOf/allOf when the top level
                    # declares no properties (upstream 0665102ce5).
                    properties = get_schema_properties(params)
                    if properties or "properties" in params:
                        return properties
                    return params
                return {}
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        if not isinstance(param_schema, dict):
            return "string"
        inferred = param_schema.get("type")
        if isinstance(inferred, list):
            inferred = next((t for t in inferred if t != "null"), "string")
        return str(inferred or "string").strip().lower()

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict
    ) -> Any:
        """Convert a raw parameter value according to the tool schema."""
        # Handle null value for any type (upstream parity).
        if param_value.lower() == "null":
            return None
        if param_name not in param_config:
            if param_config:
                logger.warning(
                    "Parsed parameter %r is not defined in the tool parameters; "
                    "returning the raw string value.",
                    param_name,
                )
            # Target extension (differs from upstream, which returns the raw
            # string): nested <parameter> leaves are looked up against the
            # top-level config and will not match, so keep the lenient
            # json.loads/literal_eval fallback for them.
            return _safe_val(param_value)
        param_type = self._get_param_type(param_config[param_name])
        if param_type in ("string", "str", "text", "varchar", "char", "enum"):
            return html.unescape(param_value)
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except Exception:
                logger.warning("Parameter %r is not an integer; degenerating to string.", param_name)
                return param_value
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                value = float(param_value)
                if value.is_integer() and "." not in param_value and "e" not in param_value.lower():
                    return int(value)
                return value
            except Exception:
                logger.warning("Parameter %r is not a float; degenerating to string.", param_name)
                return param_value
        if param_type in ("boolean", "bool", "binary"):
            lowered = param_value.lower()
            if lowered not in ("true", "false"):
                logger.warning("Parameter %r is not a boolean; degenerating to false.", param_name)
            return lowered == "true"
        if (
            param_type in ("object", "array", "arr")
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            for candidate in (param_value, html.unescape(param_value)):
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
            logger.warning(
                "Parameter %r is not valid JSON; trying Python literal fallback.", param_name
            )
        # Fallback: Python-literal style values (ast.literal_eval), with
        # invalid-escape warnings suppressed (upstream safe_literal_eval).
        try:
            return safe_literal_eval(param_value)
        except Exception:
            logger.warning(
                "Parameter %r cannot be converted via Python `ast.literal_eval()`; "
                "degenerating to string.",
                param_name,
            )
            return param_value

    def _current_tool_schema(self) -> dict:
        """JSON-schema properties of the tool call currently being parsed."""
        name = self.current_func_name or self._current_function_name
        for tool in getattr(self, "_tools", None) or []:
            if tool.function.name == name:
                return (tool.function.parameters or {}).get("properties", {}) or {}
        return {}

    def _conform_params_to_schema(self, new_params: dict) -> dict:
        """Align emitted parameter names with the tool schema: models may
        flatten nested object parameters into top-level keys. When every
        emitted key is a sub-property of a single object-typed parameter and
        none matches a top-level property, wrap them under that parameter."""
        if not new_params:
            return new_params
        props = self._current_tool_schema()
        if not props or set(new_params.keys()) <= set(props.keys()):
            return new_params
        object_params = [
            key
            for key, spec in props.items()
            if isinstance(spec, dict) and spec.get("type") == "object"
        ]
        if len(object_params) != 1:
            return new_params
        sub_props = (props[object_params[0]].get("properties") or {}) or {}
        if sub_props and set(new_params.keys()) <= set(sub_props.keys()):
            return {object_params[0]: new_params}
        return new_params

    # ------------------------------------------------------------------
    # Recursive parameter-region parser (arbitrary tag nesting)
    # ------------------------------------------------------------------
    def _parse_param_value(self, text: str, pos: int, depth: int):
        """Parse the value region after ``<parameter=name>``.

        Returns (child_items, raw_text, new_pos, closed, stopped_at):
          * child_items -- nested (name, value) pairs when the region
            contains nested ``<parameter>`` tags (container parameter);
          * raw_text -- the leaf text otherwise;
          * closed -- True when the region was terminated by its own
            ``</parameter>``;
          * stopped_at -- "closed" | "func_end" | "tool_end" | "wait".

        ``stopped_at == "wait"`` means the region is incomplete and more
        text is required; ``new_pos`` is then NOT advanced past unconsumed
        bytes, so callers must simply wait (no data is lost or duplicated).
        """
        child_items: list[tuple[str, Any]] = []
        chunks: list[str] = []
        while True:
            rest = text[pos:]
            if not rest:
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos,
                    False,
                    "wait",
                )
            if rest.startswith(self.parameter_end_token):
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos + len(self.parameter_end_token),
                    True,
                    "closed",
                )
            if rest.startswith(self.function_end_token):
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos,
                    False,
                    "func_end",
                )
            if rest.startswith(self.tool_call_end_token):
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos,
                    False,
                    "tool_end",
                )
            match = _PARAM_START_RE.match(rest)
            if match:
                name = match.group(1).strip()
                child_start = pos + match.end()
                sub_items, raw_value, new_pos, closed, stopped = self._parse_param_value(
                    text, child_start, depth + 1
                )
                if sub_items:
                    value: Any = {n: v for n, v in sub_items}
                else:
                    param_config = self._get_arguments_config(self.current_func_name)
                    value = self._convert_param_value(raw_value, name, param_config)
                child_items.append((name, value))
                if new_pos == child_start and not closed:
                    # The child is waiting for more text (or hit an end
                    # boundary without consuming it): propagate without
                    # advancing so the caller waits / finalizes.
                    return child_items, _trim_leaf_text("".join(chunks)), pos, False, stopped
                pos = new_pos
                continue
            nxt = rest.find("<")
            if nxt == -1:
                chunks.append(rest)
                pos += len(rest)
                continue
            if nxt > 0:
                chunks.append(rest[:nxt])
                pos += nxt
                continue
            # rest starts with '<': a partial terminator waits for more text;
            # a complete terminator was handled above; anything else is an
            # unknown tag inside the value -> consume through '>'.
            if any(
                t.startswith(rest)
                for t in (self.parameter_end_token, self.function_end_token, self.tool_call_end_token)
            ):
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos,
                    False,
                    "wait",
                )
            gt = rest.find(">")
            if gt == -1:
                return (
                    child_items,
                    _trim_leaf_text("".join(chunks)),
                    pos,
                    False,
                    "wait",
                )
            chunks.append(rest[: gt + 1])
            pos += gt + 1
    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)
        self._tools = tools
        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        cursor = 0
        try:
            while True:
                start = text.find(self.tool_call_start_token, cursor)
                if start == -1:
                    normal_parts.append(text[cursor:])
                    break
                normal_parts.append(text[cursor:start])
                body_start = start + len(self.tool_call_start_token)
                end = text.find(self.tool_call_end_token, body_start)
                if end == -1:
                    block = text[body_start:]
                    cursor = len(text)
                else:
                    block = text[body_start:end]
                    cursor = end + len(self.tool_call_end_token)

                func_match = self.tool_call_function_regex.search(block)
                if not func_match:
                    continue
                func_body = func_match.group(1) if func_match.group(1) else func_match.group(2)
                gt = func_body.find(">")
                if gt == -1:
                    continue
                func_name = func_body[:gt].strip()
                tool_indices = self._get_tool_indices(tools)
                if func_name not in tool_indices:
                    logger.warning("Unknown function %r in tool call; dropping.", func_name)
                    continue
                remainder = func_body[gt + 1 :]

                parsed_params: dict[str, Any] = {}
                param_config = self._get_arguments_config(func_name)
                self.current_func_name = func_name
                pos2 = 0
                while True:
                    rest = remainder[pos2:]
                    if not rest:
                        break
                    if rest.startswith(self.function_end_token) or rest.startswith(
                        self.tool_call_end_token
                    ):
                        break
                    m2 = _PARAM_START_RE.match(rest)
                    if not m2:
                        nxt = rest.find("<")
                        if nxt == -1:
                            break
                        pos2 += nxt if nxt > 0 else 1
                        continue
                    param_name = m2.group(1).strip()
                    child_items, raw_value, new_pos, closed, _ = self._parse_param_value(
                        remainder, pos2 + m2.end(), 1
                    )
                    if child_items:
                        parsed_params[param_name] = {n: v for n, v in child_items}
                    else:
                        parsed_params[param_name] = self._convert_param_value(
                            raw_value, param_name, param_config
                        )
                    pos2 = new_pos
                    if not closed:
                        break

                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=func_name,
                        parameters=json.dumps(parsed_params, ensure_ascii=False),
                    )
                )
        except Exception as e:
            logger.error("Error in detect_and_parse: %s", e)
            return StreamingParseResult(normal_text=text)
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    # ------------------------------------------------------------------
    # Streaming: cursor-based state machine (backported from upstream)
    # ------------------------------------------------------------------
    def _close_tool_args(self) -> list[ToolCallItem]:
        """Close the arguments JSON for the current tool call. If no
        <parameter> block was streamed (bare JSON body captured in
        _pending_body_text), emit its key/values first."""
        calls: list[ToolCallItem] = []
        if not self.json_started:
            body = self._pending_body_text.strip()
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self.json_started = True
            self.streamed_args_for_tool[self.current_tool_id] += "{"
            if body:
                parsed = _safe_val(body)
                if isinstance(parsed, dict):
                    parsed = self._conform_params_to_schema(parsed)
                    for k, v in parsed.items():
                        kv = (
                            f"{json.dumps(k, ensure_ascii=False)}: "
                            f"{json.dumps(v, ensure_ascii=False)}"
                        )
                        fragment = f", {kv}" if self.current_tool_param_count > 0 else kv
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=fragment
                            )
                        )
                        self.current_tool_param_count += 1
                        self.streamed_args_for_tool[self.current_tool_id] += fragment
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"][k] = v
        if not self.current_tool_args_closed:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="}"))
            self.streamed_args_for_tool[self.current_tool_id] += "}"
            self.current_tool_args_closed = True
        return calls

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
            self._tools = tools
        if _TOOL_PARSE_DEBUG:
            _dbg(f"delta={new_text!r} buf_tail={self._buffer[-60:]!r} pos={self.parsed_pos}")

        calls: list[ToolCallItem] = []
        normal_text_chunks: list[str] = []

        while True:
            current_slice = self._buffer[self.parsed_pos :]
            if not current_slice:
                break

            # 1. Tool call start
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                self._pending_body_text = ""
                _dbg("state: tool_call start")
                continue

            # 2. Function name
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle == -1:
                    break  # incomplete tag
                func_name = current_slice[len(self.tool_call_prefix) : end_angle]
                self.current_tool_id += 1
                self.current_tool_name_sent = True
                self.current_tool_param_count = 0
                self.json_started = False
                self.current_tool_args_closed = False
                self.current_func_name = func_name
                self._open_param_name = None
                self._pending_body_text = ""
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": {},
                }
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=func_name,
                        parameters="",
                    )
                )
                self.parsed_pos += end_angle + 1
                _dbg(f"state: function name={func_name!r} tool_id={self.current_tool_id}")
                continue

            # 3. Parameter: <parameter=name>value (arbitrary tag nesting)
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end == -1:
                    break  # incomplete tag
                param_name = current_slice[len(self.parameter_prefix) : name_end].strip()

                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True
                    self.streamed_args_for_tool[self.current_tool_id] += "{"

                child_items, raw_value, new_pos, closed, stopped_at = self._parse_param_value(
                    self._buffer, self.parsed_pos + name_end + 1, 1
                )
                if stopped_at == "wait":
                    # Value region incomplete: wait for more text. Remember
                    # the open parameter so flush_pending() can finalize it
                    # if generation ends without the closing tag.
                    self._open_param_name = param_name
                    self._open_param_value_start = self.parsed_pos + name_end + 1
                    _dbg(f"state: param {param_name!r} incomplete -> wait")
                    break

                if child_items:
                    # Container parameter: nested <parameter> tags were
                    # reconstructed into a dict of child values.
                    converted: Any = {n: v for n, v in child_items}
                else:
                    param_config = self._get_arguments_config(self.current_func_name)
                    converted = self._convert_param_value(
                        raw_value, param_name, param_config
                    )

                json_key_val = (
                    f"{json.dumps(param_name, ensure_ascii=False)}: "
                    f"{json.dumps(converted, ensure_ascii=False)}"
                )
                fragment = (
                    f", {json_key_val}" if self.current_tool_param_count > 0 else json_key_val
                )
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)
                )
                self.current_tool_param_count += 1
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                self.prev_tool_call_arr[self.current_tool_id]["arguments"][param_name] = converted
                self.parsed_pos = new_pos
                self._open_param_name = None
                _dbg(f"state: param {param_name!r} closed={closed} fragment={fragment!r}")

                if not closed:
                    # The region ran into </function>/</tool_call>: close the
                    # arguments JSON here; the end-token handlers below will
                    # not emit a second closing brace.
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"
                    self.current_tool_args_closed = True
                continue

            # 4. Function end: </function>
            if current_slice.startswith(self.function_end_token):
                calls.extend(self._close_tool_args())
                self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                self._pending_body_text = ""
                _dbg("state: function end (args closed)")
                continue

            # 5. Tool call end: </tool_call>. The model may omit
            # </function>: close the arguments here (bare JSON body fallback)
            # so arguments are never empty. _pending_body_text is kept for
            # flush_pending() if generation ends right after this.
            if current_slice.startswith(self.tool_call_end_token):
                if self.is_inside_tool_call and not self.current_tool_args_closed:
                    calls.extend(self._close_tool_args())
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False
                _dbg("state: tool_call end")
                continue

            # 6. Plain text / whitespace / partial tags
            next_angle = current_slice.find("<")
            if next_angle == -1:
                if self.is_inside_tool_call:
                    # Capture discarded text for the bare-JSON-body fallback.
                    self._pending_body_text += current_slice
                    if len(self._pending_body_text) > (1 << 20):
                        self._pending_body_text = self._pending_body_text[-65536:]
                else:
                    normal_text_chunks.append(current_slice)
                self.parsed_pos += len(current_slice)
                continue
            if next_angle == 0:
                possible_tags = (
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                )
                if any(tag.startswith(current_slice) for tag in possible_tags):
                    break  # potential partial tag: wait for more
                if self.is_inside_tool_call:
                    self._pending_body_text += current_slice[0]
                else:
                    normal_text_chunks.append(current_slice[0])
                self.parsed_pos += 1
                continue
            text_segment = current_slice[:next_angle]
            if self.is_inside_tool_call:
                self._pending_body_text += text_segment
            else:
                normal_text_chunks.append(text_segment)
            self.parsed_pos += next_angle

        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            # Keep the open-parameter record valid in the *sliced* buffer,
            # otherwise flush_pending() reads from a shifted offset and
            # truncates the value (differs by parsed_pos bytes).
            if self._open_param_value_start is not None:
                self._open_param_value_start -= self.parsed_pos
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks)
        if calls and _TOOL_PARSE_DEBUG:
            _dbg(f"result: calls={[(c.tool_index, c.name, c.parameters) for c in calls]}")
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def flush_pending(self) -> list[ToolCallItem]:
        """Finalize an open (unterminated) tool-call parameter at generation
        end, so pending argument fragments are never lost. Called by the
        serving layer when the finish reason arrives."""
        if self.current_tool_args_closed or self.current_tool_id < 0:
            return []
        calls: list[ToolCallItem] = []
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self.streamed_args_for_tool[self.current_tool_id] += "{"
            self.json_started = True
        param_name = self._open_param_name
        raw = ""
        if param_name and self._open_param_value_start is not None:
            raw = self._buffer[self._open_param_value_start :]
            raw = raw.replace(self.tool_call_end_token, "")
            raw = raw.replace(self.function_end_token, "")
        elif not param_name:
            raw = self._pending_body_text
        raw = raw.strip()
        if raw:
            parsed = _safe_val(raw)
            if isinstance(parsed, dict):
                parsed = self._conform_params_to_schema(parsed)
            if param_name:
                # _convert_param_value expects the *raw text* (it calls
                # .lower() / json.dumps on it); passing an already-parsed
                # dict/list crashed with AttributeError and killed the
                # stream right after the leading '{' was emitted.
                converted = self._convert_param_value(
                    raw,
                    param_name,
                    self._get_arguments_config(self.current_func_name),
                )
                kv = (
                    f"{json.dumps(param_name, ensure_ascii=False)}: "
                    f"{json.dumps(converted, ensure_ascii=False)}"
                )
            else:
                items = []
                for k, v in (parsed.items() if isinstance(parsed, dict) else []):
                    items.append(
                        f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                    )
                kv = ", ".join(items)
            fragment = f", {kv}" if self.current_tool_param_count > 0 else kv
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters=fragment))
            self.streamed_args_for_tool[self.current_tool_id] += fragment
            if param_name:
                self.prev_tool_call_arr[self.current_tool_id]["arguments"][param_name] = converted
        calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="}"))
        self.streamed_args_for_tool[self.current_tool_id] += "}"
        self.current_tool_args_closed = True
        self._open_param_name = None
        return calls

    def supports_structural_tag(self) -> bool:
        return False

    @property
    def supports_lark_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        # Previously raised NotImplementedError, which crashed
        # get_structure_tag() for strict-tool auto mode. Provide the native
        # XML envelope so strict structural_tag grammars can be built if
        # ever enabled (supports_structural_tag stays False for now; the
        # thinking-tolerant lark path in FunctionCallParser is preferred for
        # forced tool_choice with reasoning).
        return lambda name: StructureInfo(
            begin="<tool_call>\n<function=" + name + ">\n",
            end="\n</function>\n</tool_call>",
            trigger="<tool_call>",
        )

    def build_ebnf(self, tools: list[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.tool_call_start_token.replace(chr(10), "\\n"),
            individual_call_end_token=self.tool_call_end_token.replace(chr(10), "\\n"),
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
            key_value_separator='"\\n"',
        )
