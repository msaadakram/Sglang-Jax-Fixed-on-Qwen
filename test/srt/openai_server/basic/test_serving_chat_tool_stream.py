"""Regression tests for Qwen3.8 thinking + tool calling over /v1/chat/completions.

Exercises OpenAIServingChat directly with a mocked TokenizerManager, covering
the interaction of:

  * reasoning-parser ``qwen3`` (thinking ON / OFF)
  * tool-call-parser ``qwen3_coder``
  * stream=False and stream=True

Upstream references:
  * sgl-project/sglang#36537 - thinking + qwen3_coder must still emit
    structured tool_calls (and must not loop on token id 0).
  * sgl-project/sglang#29441 - no standalone empty-content SSE chunk before
    tool-call chunks (breaks AI SDK / OpenCode style consumers).
  * sgl-project/sglang#5661 - tool_calls[0].index must be numeric (0-based),
    never null.

Run with:
    python -m pytest test/srt/openai_server/basic/test_serving_chat_tool_stream.py
"""

import asyncio
import json
import re
import time
import unittest
import uuid
from unittest.mock import Mock

from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    StreamOptions,
    ToolChoice,
    ToolChoiceFuncName,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.reasoning_parser import ReasoningParser
from sgl_jax.test.tool_parser_test_config import ToolParserTestConfig as C


THINK_BLOCK = "<think>\nI should call the weather tool for Tokyo.\n</think>\n\n"


def tool_call_xml(name: str, params: dict) -> str:
    """Render the Qwen3-Coder native tool-call block."""
    out = f"<tool_call>\n<function={name}>\n"
    for key, value in params.items():
        out += f"<parameter={key}>\n{value}\n</parameter>\n"
    out += "</function>\n</tool_call>"
    return out


def get_weather_tool() -> object:
    return C.make_tool("get_weather", {"city": {"type": "string"}})


def get_time_tool() -> object:
    return C.make_tool("get_time", {"timezone": {"type": "string"}})


def search_tool() -> object:
    return C.make_tool("web_search", {"query": {"type": "string"}})


_TAG_RE = re.compile(
    r"(<think>|</think>|<tool_call>|</tool_call>|<function=|</function>|<parameter=|</parameter>|>)"
)


def _split(text: str, size: int = 4) -> list[str]:
    """Split like a detokenizer would: special tags arrive atomically as
    single tokens, plain text arrives in small multi-char pieces."""
    pieces = []
    for part in _TAG_RE.split(text):
        if not part:
            continue
        if _TAG_RE.fullmatch(part):
            pieces.append(part)
        else:
            pieces.extend(part[i : i + size] for i in range(0, len(part), size))
    return pieces


class _MockTokenizerManager:
    def __init__(self, stream_text: str):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_coder",
            enable_cache_report=False,
        )
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        async def _generate(adapted_request, raw_request=None):
            cumulative = ""
            pieces = _split(stream_text)
            for i, piece in enumerate(pieces):
                cumulative += piece
                last = i == len(pieces) - 1
                yield {
                    "index": 0,
                    "text": cumulative,
                    "meta_info": {
                        "id": rid,
                        "prompt_tokens": 32,
                        "completion_tokens": i + 1,
                        "cached_tokens": 0,
                        "finish_reason": (
                            {"type": "stop", "matched": None} if last else None
                        ),
                    },
                }

        self.generate_request = _generate


def _make_chat(stream_text: str) -> OpenAIServingChat:
    return OpenAIServingChat(_MockTokenizerManager(stream_text), Mock())


def _request(stream: bool, thinking: bool, tools=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="qwen3.8-27b",
        messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        stream=stream,
        chat_template_kwargs={"enable_thinking": thinking},
    )


def _collect_sse(chat: OpenAIServingChat, request: ChatCompletionRequest) -> list:
    async def _run():
        out = []
        async for chunk in chat._generate_chat_stream(Mock(), request, Mock()):
            out.append(chunk)
        return out

    raw = asyncio.run(_run())
    parsed = []
    for line in raw:
        assert line.startswith("data: "), f"bad SSE line: {line!r}"
        body = line[len("data: ") :].strip()
        if body == "[DONE]":
            parsed.append("[DONE]")
        else:
            parsed.append(json.loads(body))
    return parsed


def _assert_no_empty_content_chunk(chunks: list):
    """sglang#29441: never emit a standalone chunk with content=='' that has
    neither a role (first chunk) nor tool_calls."""
    for chunk in chunks:
        if chunk == "[DONE]":
            continue
        delta = chunk["choices"][0]["delta"]
        if delta.get("content") == "":
            assert delta.get("role") or delta.get("tool_calls"), (
                "empty-content SSE chunk without role/tool_calls breaks "
                "AI SDK style consumers: "
                + json.dumps(chunk)
            )


def _assert_valid_tool_stream(chunks: list, expected_names: list[str]):
    """Validate an OpenAI-compatible streamed tool-call response."""
    _assert_no_empty_content_chunk(chunks)
    assert chunks[-1] == "[DONE]", "stream must terminate with data: [DONE]"

    args_by_index: dict[int, str] = {}
    id_by_index: dict[int, str] = {}
    finish_reasons = [c for c in chunks if c != "[DONE]" and c["choices"][0]["finish_reason"]]
    assert finish_reasons, "stream never carried a finish_reason"
    final_finish = finish_reasons[-1]["choices"][0]["finish_reason"]
    assert final_finish == "tool_calls", (
        f"final finish_reason must be 'tool_calls' when tool calls were "
        f"streamed, got {final_finish!r}"
    )

    for chunk in chunks:
        if chunk == "[DONE]":
            continue
        delta = chunk["choices"][0]["delta"]
        for tc in delta.get("tool_calls") or []:
            idx = tc["index"]
            assert isinstance(idx, int), f"tool call index must be int, got {idx!r}"
            if tc.get("id"):
                if idx in id_by_index:
                    assert id_by_index[idx] == tc["id"], "tool call id must stay stable"
                id_by_index[idx] = tc["id"]
                assert tc["id"].startswith("call_")
            if tc["function"].get("name"):
                assert tc["function"]["name"] in expected_names
            frag = tc["function"].get("arguments") or ""
            args_by_index[idx] = args_by_index.get(idx, "") + frag

    assert sorted(id_by_index) == list(range(len(expected_names))), (
        f"expected indexes 0..{len(expected_names) - 1}, got {sorted(id_by_index)}"
    )
    for idx, name in enumerate(expected_names):
        args = json.loads(args_by_index[idx])  # accumulated fragments must be valid JSON
        assert isinstance(args, dict)
    return args_by_index


class TestNonStreamingToolCalls(unittest.TestCase):
    def _respond(self, text: str, thinking: bool):
        chat = _make_chat(text)
        request = _request(stream=False, thinking=thinking, tools=[get_weather_tool()])
        ret = {
            "index": 0,
            "text": text,
            "meta_info": {
                "id": "chatcmpl-test",
                "prompt_tokens": 32,
                "completion_tokens": 48,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
            },
        }
        return chat._build_chat_response(request, [ret], int(time.time())), request

    def test_tool_call_thinking_off(self):
        text = "Sure.\n\n" + tool_call_xml("get_weather", {"city": "Tokyo"})
        resp, _ = self._respond(text, thinking=False)
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        calls = choice.message.tool_calls
        assert calls and calls[0].index == 0, f"index must be 0, got {calls[0].index!r}"
        assert calls[0].id.startswith("call_")
        assert calls[0].type == "function"
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "Tokyo"}

    def test_tool_call_thinking_on(self):
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        resp, _ = self._respond(text, thinking=True)
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.reasoning_content, "thinking content must be preserved"
        calls = choice.message.tool_calls
        assert calls and calls[0].index == 0, f"index must be 0, got {calls[0].index!r}"
        assert json.loads(calls[0].function.arguments) == {"city": "Tokyo"}

    def test_two_parallel_tool_calls_indexes(self):
        text = (
            THINK_BLOCK
            + tool_call_xml("get_weather", {"city": "Tokyo"})
            + "\n"
            + tool_call_xml("get_weather", {"city": "Paris"})
        )
        resp, _ = self._respond(text, thinking=True)
        calls = resp.choices[0].message.tool_calls
        assert len(calls) == 2
        assert [c.index for c in calls] == [0, 1], (
            f"indexes must be [0, 1], got {[c.index for c in calls]}"
        )
        assert json.loads(calls[0].function.arguments) == {"city": "Tokyo"}
        assert json.loads(calls[1].function.arguments) == {"city": "Paris"}


class TestStreamingToolCalls(unittest.TestCase):
    def test_tool_stream_thinking_off(self):
        text = "Sure.\n\n" + tool_call_xml("get_weather", {"city": "Tokyo"})
        chat = _make_chat(text)
        chunks = _collect_sse(chat, _request(stream=True, thinking=False, tools=[get_weather_tool()]))
        args = _assert_valid_tool_stream(chunks, ["get_weather"])
        assert json.loads(args[0]) == {"city": "Tokyo"}

    def test_tool_stream_thinking_on(self):
        """THE critical regression: thinking ON + tools ON + streaming ON."""
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        chat = _make_chat(text)
        chunks = _collect_sse(chat, _request(stream=True, thinking=True, tools=[get_weather_tool()]))
        args = _assert_valid_tool_stream(chunks, ["get_weather"])
        assert json.loads(args[0]) == {"city": "Tokyo"}

        # reasoning content must have been streamed
        reasoning = "".join(
            c["choices"][0]["delta"].get("reasoning_content") or ""
            for c in chunks
            if c != "[DONE]"
        )
        assert "Tokyo" in reasoning or "weather" in reasoning

    def test_two_parallel_tool_stream_thinking_on(self):
        text = (
            THINK_BLOCK
            + tool_call_xml("get_weather", {"city": "Tokyo"})
            + "\n"
            + tool_call_xml("get_time", {"timezone": "JST"})
        )
        chat = _make_chat(text)
        chunks = _collect_sse(
            chat,
            _request(stream=True, thinking=True, tools=[get_weather_tool(), get_time_tool()]),
        )
        args = _assert_valid_tool_stream(chunks, ["get_weather", "get_time"])
        assert json.loads(args[0]) == {"city": "Tokyo"}
        assert json.loads(args[1]) == {"timezone": "JST"}

    def test_three_parallel_tool_stream_thinking_on(self):
        text = (
            THINK_BLOCK
            + tool_call_xml("get_weather", {"city": "Tokyo"})
            + "\n"
            + tool_call_xml("get_time", {"timezone": "JST"})
            + "\n"
            + tool_call_xml("web_search", {"query": "tokyo weather"})
        )
        chat = _make_chat(text)
        chunks = _collect_sse(
            chat,
            _request(
                stream=True,
                thinking=True,
                tools=[get_weather_tool(), get_time_tool(), search_tool()],
            ),
        )
        args = _assert_valid_tool_stream(chunks, ["get_weather", "get_time", "web_search"])
        assert json.loads(args[2]) == {"query": "tokyo weather"}


class TestStreamingTextOnly(unittest.TestCase):
    def test_reasoning_stream_no_tools(self):
        text = THINK_BLOCK + "The weather in Tokyo is sunny."
        chat = _make_chat(text)
        chunks = _collect_sse(chat, _request(stream=True, thinking=True, tools=None))
        _assert_no_empty_content_chunk(chunks)
        assert chunks[-1] == "[DONE]"
        reasoning = "".join(
            c["choices"][0]["delta"].get("reasoning_content") or ""
            for c in chunks
            if c != "[DONE]"
        )
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks if c != "[DONE]"
        )
        assert "sunny" in content
        assert "I should call" in reasoning
        final = [c for c in chunks if c != "[DONE]" and c["choices"][0]["finish_reason"]][-1]
        assert final["choices"][0]["finish_reason"] == "stop"


class TestParserPipelineIncremental(unittest.TestCase):
    """Reasoning parser -> tool detector handoff, chunk by chunk."""

    def _run_pipeline(self, text: str, tools, thinking: bool = True):
        reasoning = ReasoningParser("qwen3", stream_reasoning=True) if thinking else None
        fc = FunctionCallParser(tools, "qwen3_coder")
        calls = []
        normal = ""
        for piece in _split(text, 4):
            _, delta = reasoning.parse_stream_chunk(piece) if reasoning else (None, piece)
            if delta:
                n, new_calls = fc.parse_stream_chunk(delta)
                normal += n
                calls.extend(new_calls)
        return normal, calls, fc.detector

    def test_thinking_then_tool_call(self):
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        normal, calls, det = self._run_pipeline(text, [get_weather_tool()])
        assert det.current_tool_id == 1  # one tool completed
        assert calls[0].name == "get_weather"
        assert calls[0].tool_index == 0
        streamed = det.streamed_args_for_tool[0]
        assert json.loads(streamed) == {"city": "Tokyo"}

    def test_reasoning_end_token_split_across_chunks(self):
        """A </think> tag split across streaming increments must not leak into
        the reasoning stream nor lose the post-think tool call."""
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        reasoning = ReasoningParser("qwen3", stream_reasoning=True)
        leaked = ""
        normal = ""
        pieces = []
        for piece in _split(text, 3):
            if piece == "</think>":
                pieces.extend(["</thi", "nk>"])
            else:
                pieces.append(piece)
        for piece in pieces:
            r, n = reasoning.parse_stream_chunk(piece)
            leaked += r or ""
            normal += n or ""
        assert "</thi" not in leaked and "</think" not in leaked, (
            f"partial end tag leaked into reasoning stream: {leaked[-60:]!r}"
        )
        assert normal.lstrip().startswith("<tool_call>"), f"post-think text lost: {normal!r}"

    def test_large_string_argument_fragments(self):
        big = "x" * 8192
        text = tool_call_xml("get_weather", {"city": big})
        _, calls, det = self._run_pipeline(text, [get_weather_tool()], thinking=False)
        assert json.loads(det.streamed_args_for_tool[0]) == {"city": big}
        # incremental: the argument fragment must be emitted before the
        # closing </tool_call> is consumed, not only at finalization
        assert any(c.parameters for c in calls)

    def test_unicode_and_escapes(self):
        value = 'He said "hello" \\n 東京'
        text = tool_call_xml("get_weather", {"city": value})
        _, _, det = self._run_pipeline(text, [get_weather_tool()], thinking=False)
        assert json.loads(det.streamed_args_for_tool[0]) == {"city": value}

    def test_nested_json_parameter(self):
        nested = '{"lat": 35.6, "lon": 139.7, "tags": ["a", "b"]}'
        text = tool_call_xml("get_weather", {"city": nested})
        _, _, det = self._run_pipeline(text, [get_weather_tool()], thinking=False)
        args = json.loads(det.streamed_args_for_tool[0])
        assert args["city"] == {"lat": 35.6, "lon": 139.7, "tags": ["a", "b"]}



NEWLINE = chr(10)


class _QwenStyleTokenizer:
    """Renders messages following the Qwen3 native tool-call protocol
    (the same structure the official Qwen3 jinja template produces)."""

    bos_token_id = 1
    chat_template = "qwen3-native"  # non-None so the jinja path is taken

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs
    ):
        del tokenize, tools
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        out = ""
        for m in messages:
            role = m["role"]
            if role == "system":
                out += im_start + "system" + NEWLINE + m["content"] + im_end + NEWLINE
            elif role == "user":
                out += im_start + "user" + NEWLINE + m["content"] + im_end + NEWLINE
            elif role == "assistant":
                content = m.get("content") or ""
                block = ""
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    block += (
                        "<tool_call>" + NEWLINE
                        + json.dumps({"name": fn["name"], "arguments": fn["arguments"]})
                        + NEWLINE + "</tool_call>" + NEWLINE
                    )
                out += im_start + "assistant" + NEWLINE + content + block + im_end + NEWLINE
            elif role == "tool":
                out += (
                    im_start + "user" + NEWLINE + "<tool_response>" + NEWLINE
                    + m["content"] + NEWLINE + "</tool_response>" + im_end + NEWLINE
                )
        if add_generation_prompt:
            out += im_start + "assistant" + NEWLINE
            if kwargs.get("enable_thinking") is False:
                out += "<think>" + NEWLINE + NEWLINE + "</think>" + NEWLINE + NEWLINE
        return out

    def encode(self, text):
        return [1, 2, 3]


class TestMultiTurnToolFlow(unittest.TestCase):
    """user -> assistant tool_call -> tool result -> assistant final answer,
    including the two-parallel-tool-calls variant."""

    def _prompt(self, messages):
        tm = Mock()
        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_coder",
            enable_cache_report=False,
        )
        tm.tokenizer = _QwenStyleTokenizer()
        tm.mm_processor = None
        chat = OpenAIServingChat(tm, Mock())
        chat.template_manager.chat_template_name = None
        chat.template_manager.jinja_template_content_format = None
        request = ChatCompletionRequest(
            model="qwen3.8-27b",
            messages=messages,
            tools=[get_weather_tool()],
            stream=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        adapted, _ = chat._convert_to_internal_request(request)
        return adapted.text

    def test_single_tool_multi_turn(self):
        prompt = self._prompt(
            [
                {"role": "user", "content": "Weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "22C, sunny", "tool_call_id": "call_abc"},
                {"role": "user", "content": "Summarize that."},
            ]
        )
        expected_call = (
            "<tool_call>" + NEWLINE
            + json.dumps({"name": "get_weather", "arguments": {"city": "Tokyo"}})
            + NEWLINE + "</tool_call>"
        )
        assert expected_call in prompt
        assert "<tool_response>" + NEWLINE + "22C, sunny" in prompt
        assert "<|im_start|>user" + NEWLINE + "Summarize that." in prompt
        assert prompt.endswith("<|im_start|>assistant" + NEWLINE)

    def test_two_parallel_tools_multi_turn(self):
        prompt = self._prompt(
            [
                {"role": "user", "content": "Weather in Tokyo and Osaka?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Osaka"}),
                            },
                        },
                    ],
                },
                {"role": "tool", "content": "22C sunny", "tool_call_id": "call_a"},
                {"role": "tool", "content": "18C cloudy", "tool_call_id": "call_b"},
                {"role": "user", "content": "Compare them."},
            ]
        )
        assert '"city": "Tokyo"' in prompt and '"city": "Osaka"' in prompt
        assert "22C sunny" in prompt and "18C cloudy" in prompt
        assert prompt.count("<tool_response>") == 2
        assert prompt.endswith("<|im_start|>assistant" + NEWLINE)


class TestStreamOptionsUsage(unittest.TestCase):
    def test_include_usage_stream(self):
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        chat = _make_chat(text)
        request = _request(stream=True, thinking=True, tools=[get_weather_tool()])
        request.stream_options = StreamOptions(include_usage=True)
        chunks = _collect_sse(chat, request)
        assert chunks[-1] == "[DONE]"
        usage = [c for c in chunks if c != "[DONE]" and c.get("usage")]
        assert usage, "usage chunk missing"
        assert usage[-1]["choices"] == []  # OpenAI spec: usage chunk has no choices
        finishes = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c != "[DONE]" and c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finishes[-1] == "tool_calls"


TOOLS = [C.make_tool("get_weather", {"city": {"type": "string"}})]


NEWLINE = chr(10)


class TestFinalizationAndProtocolEdges(unittest.TestCase):
    """Adversarial edge cases found during re-verification."""

    def _stream(self, events):
        tm = Mock()
        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_coder",
            enable_cache_report=False,
        )

        async def gen(_a, _r=None):
            for e in events:
                yield e

        tm.generate_request = gen
        return OpenAIServingChat(tm, Mock())

    @staticmethod
    def _ev(idx, text, finish, rid="x"):
        return {
            "index": idx,
            "text": text,
            "meta_info": {
                "id": rid,
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": finish,
            },
        }

    def _scan(self, chunks):
        args, finish, reasoning = {}, None, ""
        for c in chunks:
            if c == "[DONE]":
                continue
            ch = c["choices"][0]
            for t in ch["delta"].get("tool_calls") or []:
                args[t["index"]] = args.get(t["index"], "") + (t["function"].get("arguments") or "")
            if ch["finish_reason"]:
                finish = ch["finish_reason"]
            reasoning += ch["delta"].get("reasoning_content") or ""
        return args, finish, reasoning

    def test_finish_event_with_tool_tail_in_same_delta(self):
        """Finish event whose delta carries the whole tool-call tail: the
        finalization must flush remaining args exactly once (no doubled
        closing brace, no duplicated arguments)."""
        events = [self._ev(0, "", None)]
        cum = ""
        for i, piece in enumerate(
            ["<think>", "r", "</think>", "N", "<tool_call>", "<function=get_weather>",
             "<parameter=city>", "Tokyo", "</parameter>", "</function>", "</tool_call>"]
        ):
            cum += piece
            last = piece == "</tool_call>"
            events.append(
                self._ev(0, cum, {"type": "stop", "matched": None} if last else None)
            )
        chat = self._stream(events)
        chunks = _collect_sse(chat, _request(stream=True, thinking=True, tools=TOOLS))
        args, finish, _ = self._scan(chunks)
        assert json.loads(args[0]) == {"city": "Tokyo"}
        assert finish == "tool_calls"
        assert chunks[-1] == "[DONE]"

    def test_finish_event_in_separate_empty_delta(self):
        """Real detokenizer behavior: </tool_call> streams first, the finish
        event arrives separately with an empty delta."""
        events = [self._ev(0, "", None)]
        cum = ""
        for piece in ["<think>", "r", "</think>", "N", "<tool_call>",
                      "<function=get_weather>", "<parameter=city>", "Tokyo",
                      "</parameter>", "</function>", "</tool_call>"]:
            cum += piece
            events.append(self._ev(0, cum, None))
        events.append(self._ev(0, cum, {"type": "stop", "matched": None}))
        chat = self._stream(events)
        chunks = _collect_sse(chat, _request(stream=True, thinking=True, tools=TOOLS))
        args, finish, _ = self._scan(chunks)
        assert json.loads(args[0]) == {"city": "Tokyo"}
        assert finish == "tool_calls"

    def test_every_choice_gets_finish_chunk_n2(self):
        """n=2 stream: EACH choice must receive its own finish_reason chunk
        (the finish chunk used to be emitted once, after the loop, using the
        last event's variables - dropping it for all but the last choice)."""
        events = []
        for idx in (0, 1):
            cum = ""
            events.append(self._ev(idx, "", None, rid=f"n{idx}"))
            for piece in ["<think>", "r", "</think>", "N", "<tool_call>",
                          "<function=get_weather>", "<parameter=city>",
                          "Tokyo" if idx == 0 else "Paris",
                          "</parameter>", "</function>", "</tool_call>"]:
                cum += piece
                last = piece == "</tool_call>"
                events.append(self._ev(idx, cum,
                                       {"type": "stop", "matched": None} if last else None,
                                       rid=f"n{idx}"))
        chat = self._stream(events)
        chunks = _collect_sse(chat, _request(stream=True, thinking=True, tools=TOOLS))
        finishes = {}
        args = {}
        for c in chunks:
            if c == "[DONE]":
                continue
            ch = c["choices"][0]
            if ch["finish_reason"]:
                finishes[ch["index"]] = ch["finish_reason"]
            for t in ch["delta"].get("tool_calls") or []:
                args.setdefault(ch["index"], {})
                i = t["index"]
                args[ch["index"]][i] = args[ch["index"]].get(i, "") + (t["function"].get("arguments") or "")
        assert finishes == {0: "tool_calls", 1: "tool_calls"}
        assert json.loads(args[0][0]) == {"city": "Tokyo"}
        assert json.loads(args[1][0]) == {"city": "Paris"}
        assert chunks[-1] == "[DONE]"

    def test_stream_reasoning_false_thinking_plus_tools(self):
        """stream_reasoning=false: reasoning is buffered; the <think> token
        must not leak into reasoning_content and the tool call must still
        produce valid arguments."""
        events = [self._ev(0, "", None)]
        cum = ""
        for i, piece in enumerate(["<think>", "secret plan", "</think>", "N", "<tool_call>",
                                   "<function=get_weather>", "<parameter=city>", "Tokyo",
                                   "</parameter>", "</function>", "</tool_call>"]):
            cum += piece
            last = piece == "</tool_call>"
            events.append(self._ev(0, cum,
                                   {"type": "stop", "matched": None} if last else None))
        chat = self._stream(events)
        request = _request(stream=True, thinking=True, tools=TOOLS)
        request.stream_reasoning = False
        chunks = _collect_sse(chat, request)
        args, finish, reasoning = self._scan(chunks)
        assert "<think>" not in reasoning
        assert "secret plan" in reasoning
        assert json.loads(args[0]) == {"city": "Tokyo"}
        assert finish == "tool_calls"


NEWLINE = chr(10)


class TestControlTagSplitAcrossEvents(unittest.TestCase):
    """A control tag split across detokenizer events must never be lost.

    Stream-only failure mode: a detokenizer event that is a strict prefix of
    <think>/<tool_call> (e.g. a lone "<") is held by the reasoning parser's
    prefix check; the post-reasoning path used to return only the new text,
    silently dropping the held fragment -- a following <tool_call> then lost
    its start token and the tool call was never detected in streaming mode.
    """

    TEXT = (
        "<think>" + NEWLINE + "reason" + NEWLINE + "</think>" + NEWLINE + NEWLINE
        + "<tool_call>" + NEWLINE + "<function=get_weather>" + NEWLINE
        + "<parameter=city>" + NEWLINE + "Tokyo" + NEWLINE + "</parameter>"
        + NEWLINE + "</function>" + NEWLINE + "</tool_call>"
    )

    def _run_chain(self, tag: str, split_at: int):
        pre, post = self.TEXT.split(tag, 1)
        events = [e for e in [pre, tag[:split_at], tag[split_at:], post] if e]
        reasoning = ReasoningParser("qwen3", stream_reasoning=True)
        fc = FunctionCallParser(TOOLS, "qwen3_coder")
        names, args_fragments = [], ""
        for ev in events:
            _, delta = reasoning.parse_stream_chunk(ev)
            if delta:
                _, calls = fc.parse_stream_chunk(delta)
                for c in calls:
                    if c.name:
                        names.append(c.name)
                    args_fragments += c.parameters or ""
        return names, args_fragments

    def test_think_end_tag_split_at_every_position(self):
        for k in range(1, 8):
            with self.subTest(tag="</think>", split=k):
                names, frag = self._run_chain("</think>", k)
                self.assertEqual(names, ["get_weather"])
                self.assertEqual(json.loads(frag), {"city": "Tokyo"})

    def test_tool_call_tag_split_at_every_position(self):
        for k in range(1, 10):
            with self.subTest(tag="<tool_call>", split=k):
                names, frag = self._run_chain("<tool_call>", k)
                self.assertEqual(names, ["get_weather"])
                self.assertEqual(json.loads(frag), {"city": "Tokyo"})

    def test_lone_lt_character_after_reasoning_is_preserved(self):
        """A "<" in post-reasoning text (e.g. "5 < 10") must not be dropped."""
        reasoning = ReasoningParser("qwen3", stream_reasoning=True)
        out = ""
        for piece in ["<think>", "r", "</think>", "5 <", " 10 and 3 <", " 4"]:
            _, delta = reasoning.parse_stream_chunk(piece)
            out += delta or ""
        self.assertEqual(out, "5 < 10 and 3 < 4")


NEWLINE = chr(10)


class TestForcedToolChoice(unittest.TestCase):
    """Forced tool_choice + thinking previously produced a 1-token response:
    the json_schema constraint masked <think> at step 0, so at temperature 0
    the model deterministically emitted the EOS/stop token
    (completion_tokens=1, matched_stop=<stop id>, finish_reason=stop).
    Forced tool_choice with thinking now uses a thinking-tolerant structural
    tag (free reasoning, then the native <tool_call> structure)."""

    def _constraint(self, tool_choice, thinking):
        p = FunctionCallParser(TOOLS, "qwen3_coder")
        return p.get_structure_constraint(tool_choice, thinking=thinking)

    def test_constraint_selection_matrix(self):
        self.assertIsNone(self._constraint("auto", thinking=True))
        self.assertEqual(self._constraint("required", thinking=True)[0], "structural_tag")
        self.assertEqual(
            self._constraint(
                ToolChoice(type="function", function=ToolChoiceFuncName(name="get_weather")),
                thinking=True,
            )[0],
            "structural_tag",
        )
        # thinking disabled keeps the historical json_schema path
        self.assertEqual(self._constraint("required", thinking=False)[0], "json_schema")

    def test_thinking_tag_carries_lark_grammars(self):
        kind, value = self._constraint("required", thinking=True)
        d = value.model_dump(by_alias=True)
        self.assertIn("lark_grammars", d)
        self.assertIn("struct_tag", d["lark_grammars"])
        self.assertIn("tag_body", d["lark_grammars"])
        main = d["lark_grammars"]["struct_tag"]
        # bare special-token references (quoted literals cannot match them)
        self.assertIn("think_part: <think>", main)
        self.assertIn("</think>", main)
        self.assertIn("@tag_body", main)

    def test_forced_tool_call_with_thinking_non_stream(self):
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        chat = _make_chat(text)
        request = _request(stream=False, thinking=True, tools=TOOLS)
        request.tool_choice = "required"
        ret = {
            "index": 0, "text": text,
            "meta_info": {"id": "c", "prompt_tokens": 8, "completion_tokens": 40,
                          "cached_tokens": 0,
                          "finish_reason": {"type": "stop", "matched": None}},
        }
        resp = chat._build_chat_response(request, [ret], int(time.time()))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNotNone(choice.message.tool_calls)
        self.assertEqual(choice.message.tool_calls[0].index, 0)
        self.assertEqual(choice.message.tool_calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(choice.message.tool_calls[0].function.arguments), {"city": "Tokyo"})
        self.assertTrue(choice.message.reasoning_content)

    def test_forced_tool_call_with_thinking_stream(self):
        text = THINK_BLOCK + tool_call_xml("get_weather", {"city": "Tokyo"})
        chat = _make_chat(text)
        request = _request(stream=True, thinking=True, tools=TOOLS)
        request.tool_choice = "required"
        chunks = _collect_sse(chat, request)
        args, finish, done = {}, None, False
        for c in chunks:
            if c == "[DONE]":
                done = True
                continue
            ch = c["choices"][0]
            for t in ch["delta"].get("tool_calls") or []:
                args[t["index"]] = args.get(t["index"], "") + (t["function"].get("arguments") or "")
            if ch["finish_reason"]:
                finish = ch["finish_reason"]
        self.assertEqual(json.loads(args[0]), {"city": "Tokyo"})
        self.assertEqual(finish, "tool_calls")
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
