"""Regression tests for Qwen3CoderDetector flush/end-of-stream + backported helpers.

Locks in the backport of upstream sglang commits 0665102ce5 / c20c48b8fd
(nested tool-call arguments truncated in streaming mode):

  * EOS/stop-token arriving mid-<parameter> (unclosed value): flush_pending()
    must reconstruct the full JSON arguments (the create_event failure class)
    instead of crashing on `dict.lower` or emitting a shifted/truncated tail.
  * Buffer-slice offset accounting: an open parameter's value start must stay
    valid when the streaming buffer is compacted (parsed_pos slice).
  * Backported utils helpers: safe_literal_eval (invalid-escape warnings
    suppressed, thread-safe) and get_schema_properties (anyOf/oneOf/allOf
    descent).
  * 13 pre-existing behaviors that must not regress: tag emission order,
    nested parameter tags (h1/h2), JSON values with `>` and braces, tags split
    across chunks, escaped quotes, unicode, multiple sequential calls,
    state reset between calls, thinking/tool transitions, bare-JSON bodies,
    array leaves, anyOf schemas, null handling.

Run with:
    python test/srt/function_call/test_qwen3_coder_flush_regression.py
"""

import json
import threading
import unittest
import warnings

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sgl_jax.srt.function_call.utils import get_schema_properties, safe_literal_eval

NL = "\n"


def make_tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": properties},
        ),
    )


EVENT_TOOL = make_tool(
    "create_event",
    {
        "event": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "location": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
            },
        }
    },
)

EVENT_EV = {"title": "Team Meeting", "location": "Lahore", "attendees": ["Ali", "Sara"]}
BASH_TOOL = make_tool("execute_bash", {"command": {"type": "string"}})


def tool_call_text(body: str, func: str = "create_event") -> str:
    return f"<tool_call>{NL}<function={func}>{NL}{body}{NL}</tool_call>"


def stream(text: str, tools: list, chunk_size: int | None = None):
    """Feed text through the streaming detector; return (detector, calls, normal).

    chunk_size=None feeds the whole text in one increment (whole-chunk mode);
    an int feeds fixed-size chunks (token-ish mode); 1 feeds char-by-char.
    """
    d = Qwen3CoderDetector()
    calls, normal = [], ""
    chunks = [text] if chunk_size is None else [
        text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
    ]
    for chunk in chunks:
        r = d.parse_streaming_increment(chunk, tools)
        calls.extend(r.calls)
        normal += r.normal_text
    return d, calls, normal


def joined_args(calls) -> str:
    return "".join(c.parameters for c in calls if c.name is None)


def flush_args(d) -> list:
    return d.flush_pending()


class TestSafeLiteralEval(unittest.TestCase):
    """Backported helper: ast.literal_eval with warnings suppressed."""

    def test_01_basic_literals(self):
        self.assertEqual(safe_literal_eval("{'a': 1}"), {"a": 1})
        self.assertEqual(safe_literal_eval("(1, 2)"), (1, 2))
        self.assertEqual(safe_literal_eval("[1, 2]"), [1, 2])
        self.assertEqual(safe_literal_eval("42"), 42)

    def test_02_invalid_escape_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> raise
            self.assertEqual(safe_literal_eval(r"'\d+'"), r"\d+")

    def test_03_invalid_escape_thread_safe(self):
        # catch_warnings mutates global state; concurrent calls must all work.
        results, errors = [], []

        def worker():
            for _ in range(50):
                try:
                    results.append(safe_literal_eval(r"'\w+\s\d'"))
                except Exception as e:  # pragma: no cover
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 400)

    def test_04_rejects_non_literal(self):
        with self.assertRaises(Exception):
            safe_literal_eval("__import__('os')")


class TestGetSchemaProperties(unittest.TestCase):
    """Backported helper: properties lookup with anyOf/oneOf/allOf descent."""

    def test_05_direct_properties(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertEqual(get_schema_properties(schema)["a"]["type"], "string")

    def test_06_anyof_descent(self):
        schema = {
            "type": "object",
            "anyOf": [
                {"properties": {"sql": {"type": "string"}}},
                {"properties": {"rows": {"type": "array"}}},
            ],
        }
        props = get_schema_properties(schema)
        self.assertIn("sql", props)
        self.assertIn("rows", props)

    def test_07_oneof_allof_and_non_dict(self):
        one = get_schema_properties({"oneOf": [{"properties": {"x": 1}}]})
        all_ = get_schema_properties({"allOf": [{"properties": {"y": 2}}]})
        self.assertEqual(set(one), {"x"})
        self.assertEqual(set(all_), {"y"})
        self.assertEqual(get_schema_properties("not-a-dict"), {})
        self.assertEqual(get_schema_properties(None), {})

    def test_08_first_branch_wins_on_conflict(self):
        schema = {
            "anyOf": [
                {"properties": {"k": "branch1"}},
                {"properties": {"k": "branch2"}},
            ]
        }
        self.assertEqual(get_schema_properties(schema)["k"], "branch1")


class TestFlushMidParameter(unittest.TestCase):
    """The core fix: EOS/stop arrives before </parameter> (create_event)."""

    UNCLOSED = (
        f"<parameter=event>{NL}"
        '{"title": "Team Meeting", "location": "Lahore", '
        '"attendees": ["Ali", "Sara"]}'
    )

    def _run(self, text: str, chunking):
        d, calls, _ = stream(text, [EVENT_TOOL], chunking)
        flush = flush_args(d)
        all_calls = calls + flush
        args = joined_args(all_calls)
        self.assertTrue(
            args.startswith("{") and args.endswith("}"),
            f"args not braced: {args!r}",
        )
        parsed = json.loads(args)
        self.assertEqual(parsed["event"], EVENT_EV)
        self.assertEqual(
            d.prev_tool_call_arr[0]["arguments"]["event"],
            EVENT_EV,
            "prev_tool_call_arr must carry the reconstructed nested object",
        )
        self.assertTrue(d.current_tool_args_closed)
        # streamed_args must mirror the emitted fragments
        self.assertEqual(
            d.streamed_args_for_tool[0],
            args,
            "streamed_args_for_tool must equal reconstructed JSON",
        )

    def test_09_flush_char_streaming(self):
        self._run(tool_call_text(self.UNCLOSED), 1)

    def test_10_flush_whole_chunk(self):
        self._run(tool_call_text(self.UNCLOSED), None)

    def test_11_flush_token_chunks(self):
        self._run(tool_call_text(self.UNCLOSED), 7)

    def test_12_flush_scalar_param(self):
        # Unclosed *scalar* (string) parameter, e.g. execute_bash command.
        text = tool_call_text("<parameter=command>pwd && ls", func="execute_bash")
        d, calls, _ = stream(text, [BASH_TOOL], 1)
        flush = flush_args(d)
        args = joined_args(calls + flush)
        self.assertEqual(json.loads(args), {"command": "pwd && ls"})

    def test_13_flush_after_complete_params_is_noop(self):
        # Closed tool call: flush must emit nothing.
        text = (
            tool_call_text(
                f'<parameter=event>{NL}{json.dumps(EVENT_EV)}{NL}</parameter>'
            )
            + f"{NL}</function>"
        )
        d, calls, _ = stream(text, [EVENT_TOOL], 1)
        self.assertEqual(joined_args(calls), json.dumps({"event": EVENT_EV}))
        self.assertEqual(flush_args(d), [])

    def test_14_flush_no_open_tool(self):
        d, _, _ = stream("just chatting", [EVENT_TOOL], 1)
        self.assertEqual(flush_args(d), [])


class TestNestedTagEmission(unittest.TestCase):
    """h1-h4: nested <parameter> tags must reconstruct full arguments."""

    def _assert_both_chunkings(self, body, expected=None):
        if expected is None:
            expected = {"event": EVENT_EV}
        text = tool_call_text(body) + f"{NL}</function>"
        for cs in (1, 6):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [EVENT_TOOL], cs)
                flush = flush_args(d)
                self.assertEqual(json.loads(joined_args(calls + flush)), expected)
                self.assertEqual(flush, [], "closed call must not be flushed again")

    def test_15_h1_nested_parameter_tags(self):
        self._assert_both_chunkings(
            f"<parameter=event>{NL}"
            f"<parameter=title>{NL}Team Meeting{NL}</parameter>{NL}"
            f"<parameter=location>{NL}Lahore{NL}</parameter>{NL}"
            f'<parameter=attendees>{NL}["Ali", "Sara"]{NL}</parameter>{NL}'
            f"</parameter>"
        )

    def test_16_h2_nested_outer_unclosed(self):
        self._assert_both_chunkings(
            f"<parameter=event>{NL}"
            f"<parameter=title>{NL}Team Meeting{NL}</parameter>{NL}"
            f"<parameter=location>{NL}Lahore{NL}</parameter>{NL}"
            f'<parameter=attendees>{NL}["Ali", "Sara"]{NL}</parameter>'
        )

    def test_17_h3_json_value_with_gt_and_braces(self):
        value = {
            "title": "a > b",
            "meta": {"ok": True},
            "attendees": ["Ali", "Sara"],
        }
        self._assert_both_chunkings(
            f"<parameter=event>{NL}{json.dumps(value)}{NL}</parameter>",
            {"event": value},
        )

    def test_18_h4_parameter_tag_split_across_chunks(self):
        body = (
            "<par" + "ameter=" + "ev" + "ent>" + NL
            + json.dumps(EVENT_EV) + NL + "</parameter>"
        )
        text = tool_call_text(body) + f"{NL}</function>"
        for cs in (1, 6):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [EVENT_TOOL], cs)
                self.assertEqual(
                    json.loads(joined_args(calls + flush_args(d))),
                    {"event": EVENT_EV},
                )


class TestChunkSplitMarkers(unittest.TestCase):
    """Markers split across chunk boundaries must never leak into output."""

    def test_19_no_marker_leak(self):
        text = (
            "before"
            + tool_call_text(
                f'<parameter=event>{NL}{json.dumps(EVENT_EV)}{NL}</parameter>'
            )
            + f"{NL}</function>after"
        )
        for cs in (1, 2, 3, 5, 8):
            with self.subTest(chunk=cs):
                d, calls, normal = stream(text, [EVENT_TOOL], cs)
                self.assertEqual(
                    json.loads(joined_args(calls + flush_args(d))),
                    {"event": EVENT_EV},
                )
                self.assertNotIn("<tool_call>", normal)
                self.assertNotIn("<parameter=", normal)
                self.assertNotIn("</parameter>", normal)
                self.assertNotIn("<function=", normal)


class TestEscapedQuotesAndUnicode(unittest.TestCase):
    def test_20_escaped_quotes_in_json(self):
        ev = {"title": 'say "hi"', "location": "a\\b", "attendees": []}
        text = (
            tool_call_text(f"<parameter=event>{NL}{json.dumps(ev)}{NL}</parameter>")
            + f"{NL}</function>"
        )
        for cs in (1, None):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [EVENT_TOOL], cs)
                self.assertEqual(
                    json.loads(joined_args(calls + flush_args(d))), {"event": ev}
                )

    def test_21_unicode_content(self):
        ev = {"title": "会议 🎉", "location": "Lahore", "attendees": ["Sara"]}
        text = (
            tool_call_text(f"<parameter=event>{NL}{json.dumps(ev)}{NL}</parameter>")
            + f"{NL}</function>"
        )
        for cs in (1, None):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [EVENT_TOOL], cs)
                self.assertEqual(
                    json.loads(joined_args(calls + flush_args(d))), {"event": ev}
                )


class TestMultipleCallsAndStateReset(unittest.TestCase):
    def test_22_two_sequential_calls_distinct_indices(self):
        text = (
            tool_call_text("<parameter=command>pwd</parameter>", "execute_bash")
            + f"{NL}</function>{NL}"
            + "middle text"
            + NL
            + tool_call_text(
                f'<parameter=event>{NL}{{"title": "X"}}{NL}</parameter>', "create_event"
            )
            + f"{NL}</function>"
        )
        for cs in (1, None):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [BASH_TOOL, EVENT_TOOL], cs)
                by_index = {}
                for c in calls:
                    if c.name is not None:
                        continue
                    by_index.setdefault(c.tool_index, "")
                    by_index[c.tool_index] += c.parameters
                self.assertEqual(json.loads(by_index[0]), {"command": "pwd"})
                self.assertEqual(json.loads(by_index[1]), {"event": {"title": "X"}})
                self.assertEqual(d.current_tool_id, 1)

    def test_23_state_reset_after_close_then_new_call(self):
        # Second call after a fully closed first one must start fresh JSON.
        text = (
            tool_call_text("<parameter=command>ls</parameter>", "execute_bash")
            + f"{NL}</function>{NL}"
            + tool_call_text("<parameter=command>pwd</parameter>", "execute_bash")
            + f"{NL}</function>"
        )
        d, calls, _ = stream(text, [BASH_TOOL], 1)
        args = {}
        for c in calls:
            if c.name is None:
                args.setdefault(c.tool_index, "")
                args[c.tool_index] += c.parameters
        self.assertEqual(json.loads(args[0]), {"command": "ls"})
        self.assertEqual(json.loads(args[1]), {"command": "pwd"})

    def test_24_thinking_tool_thinking_tool_transition(self):
        text = (
            "<think>reason one</think>"
            + tool_call_text("<parameter=command>pwd</parameter>", "execute_bash")
            + f"{NL}</function>{NL}"
            + "<think>reason two</think>"
            + tool_call_text("<parameter=command>ls -la</parameter>", "execute_bash")
            + f"{NL}</function>"
        )
        d, calls, _ = stream(text, [BASH_TOOL], 1)
        args = {}
        for c in calls:
            if c.name is None:
                args.setdefault(c.tool_index, "")
                args[c.tool_index] += c.parameters
        self.assertEqual(json.loads(args[0]), {"command": "pwd"})
        self.assertEqual(json.loads(args[1]), {"command": "ls -la"})


class TestSchemaDrivenConversion(unittest.TestCase):
    def test_25_bare_json_body_closed(self):
        # Whole tool-call body is bare JSON (no <parameter> tags).
        text = tool_call_text(json.dumps({"event": EVENT_EV})) + f"{NL}</function>"
        for cs in (1, None):
            with self.subTest(chunk=cs):
                d, calls, _ = stream(text, [EVENT_TOOL], cs)
                self.assertEqual(
                    json.loads(joined_args(calls + flush_args(d))),
                    {"event": EVENT_EV},
                )

    def test_26_array_leaf_top_level(self):
        tool = make_tool(
            "create_event",
            {"attendees": {"type": "array", "items": {"type": "string"}}},
        )
        text = (
            tool_call_text('<parameter=attendees>["Ali", "Sara"]</parameter>')
            + f"{NL}</function>"
        )
        d, calls, _ = stream(text, [tool], 1)
        self.assertEqual(
            json.loads(joined_args(calls + flush_args(d))),
            {"attendees": ["Ali", "Sara"]},
        )

    def test_27_anyof_schema_both_paths(self):
        tool = Tool(
            type="function",
            function=Function(
                name="query_db",
                description="db",
                parameters={
                    "type": "object",
                    "anyOf": [
                        {"properties": {"sql": {"type": "string"}}},
                        {
                            "properties": {
                                "rows": {"type": "array", "items": {"type": "integer"}}
                            }
                        },
                    ],
                },
            ),
        )
        # Non-streaming path
        r = Qwen3CoderDetector().detect_and_parse(
            tool_call_text("<parameter=sql>SELECT 1</parameter>", "query_db")
            + f"{NL}</function>",
            [tool],
        )
        self.assertEqual(json.loads(r.calls[0].parameters), {"sql": "SELECT 1"})
        # Streaming path
        d, calls, _ = stream(
            tool_call_text("<parameter=rows>[1, 2]</parameter>", "query_db")
            + f"{NL}</function>",
            [tool],
            1,
        )
        self.assertEqual(
            json.loads(joined_args(calls + flush_args(d))), {"rows": [1, 2]}
        )

    def test_28_null_value(self):
        tool = make_tool("create_event", {"event": {"type": "object"}})
        text = tool_call_text("<parameter=event>null</parameter>") + f"{NL}</function>"
        d, calls, _ = stream(text, [tool], 1)
        self.assertEqual(json.loads(joined_args(calls + flush_args(d))), {"event": None})

    def test_29_nonstream_unclosed_nested_param(self):
        # detect_and_parse parity for the unclosed-nested case.
        r = Qwen3CoderDetector().detect_and_parse(
            tool_call_text('<parameter=event>{"title": "T"}') + f"{NL}",
            [EVENT_TOOL],
        )
        self.assertEqual(json.loads(r.calls[0].parameters), {"event": {"title": "T"}})

    def test_30_thinking_forced_tag_compiles_and_accepts_tool_call(self):
        """Thinking-tolerant structural tag must compile in llguidance and
        accept a full reasoning + tool call (the forced tool_choice path).

        Regression: the tag previously referenced  thinking/ response as
        bare special-token names, which llguidance rejects ("unknown special
        token") -> INVALID_GRAMMAR_OBJ -> no constraint -> tool_choice=required
        silently chatted instead of forcing a tool call.
        """
        try:
            import importlib.util

            if importlib.util.find_spec("llguidance") is None:
                self.skipTest("llguidance not installed")
        except (ImportError, ValueError):
            self.skipTest("llguidance not importable")

        import os

        from llguidance import LLTokenizer

        from sgl_jax.srt.constrained.llguidance_backend import GuidanceBackend
        from sgl_jax.srt.entrypoints.openai.protocol import ToolChoice
        from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser

        tok_path = os.environ.get("SGL_JAX_TEST_TOKENIZER_JSON")
        if not tok_path or not os.path.exists(tok_path):
            self.skipTest("SGL_JAX_TEST_TOKENIZER_JSON not set")

        tok = LLTokenizer(tok_path)
        tool = make_tool(
            "get_weather",
            {"city": {"type": "string"}},
        )
        tc = ToolChoice.model_validate(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        p = FunctionCallParser([tool], "qwen3_coder")
        c = p.get_structure_constraint(tc, thinking=True)
        self.assertIsNotNone(c)
        kind, value = c
        self.assertEqual(kind, "structural_tag")
        key = value.model_dump_json(by_alias=True)
        g = GuidanceBackend(tokenizer=tok).dispatch_structural_tag(key)
        self.assertTrue(hasattr(g, "ll_matcher"), "thinking tag compiled to INVALID_GRAMMAR_OBJ")
        m = g.ll_matcher
        self.assertFalse(m.is_error(), m.get_error())
        # Free text (reasoning) + full forced tool call must be accepted.
        full = (
            "Let me think about the weather\n\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
            "Lahore\n</parameter>\n</function>\n</tool_call>"
        )
        for i in tok.greedy_tokenize(full):
            m.consume_token(i)
            if m.is_error():
                self.fail(
                    f"grammar rejected token {tok.decode_str([i])[:30]!r}: {m.get_error()}"
                )
            if m.is_stopped():
                break


class TestTC45ForcedCompliance(unittest.TestCase):
    """TC-45: tool_choice=required must use a forcing grammar, not the
    permissive thinking-tolerant tag.

    The thinking-tolerant structural_tag allows indefinite free-text reasoning
    (lazy TAG_TEXT) and only forbids EOS without a <tool_call>. With the bench
    SYSTEM_PROMPT ("If you can answer directly ... do so without calling a
    tool"), the model loops in thinking ("7 times 8 is 56" x500) and never
    emits the trigger -> missing_step fail. Forced tool_choice must therefore
    disable thinking FIRST so the constraint is the forcing from-token-0
    native EBNF, which rejects free text outright.
    """

    def _make_serving(self, reasoning_parser="qwen3", tool_parser="qwen3_coder"):
        from unittest.mock import MagicMock

        from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        tok_mgr = MagicMock()
        tok_mgr.server_args.reasoning_parser = reasoning_parser
        tok_mgr.server_args.tool_call_parser = tool_parser
        tmpl_mgr = MagicMock()
        tmpl_mgr.chat_template_name = None
        tok_mgr.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        tok_mgr.tokenizer.bos_token_id = 1
        tok_mgr.mm_processor = None
        tmpl_mgr.jinja_template_content_format = "openai"
        return OpenAIServingChat(tok_mgr, tmpl_mgr)

    def _make_request(self, tool_choice="required", chat_kwargs=None):
        from sgl_jax.srt.entrypoints.openai.protocol import ChatCompletionRequest

        calc = make_tool("calculator", {"expression": {"type": "string"}})
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "What is 7 times 8?"}],
                "tools": [calc.model_dump()],
                "tool_choice": tool_choice,
                "temperature": 0.0,
                "chat_template_kwargs": dict(chat_kwargs) if chat_kwargs else None,
            }
        )
        return req

    def test_31_forced_required_disables_thinking_and_forces_native_ebnf(self):
        for rp, tp, kwargs in [
            ("qwen3", "qwen3_coder", None),
            ("qwen3", "qwen3_coder", {"enable_thinking": True}),
            (None, "qwen3_coder", None),
            (None, "qwen25", None),
            ("qwen3", "qwen25", None),
        ]:
            with self.subTest(reasoning_parser=rp, tool_parser=tp, kwargs=kwargs):
                svc = self._make_serving(rp, tp)
                req = self._make_request("required", kwargs)
                res = svc._process_messages(req, is_multimodal=False)
                self.assertIsNotNone(res.tool_call_constraint)
                # Qwen forced tool_choice uses the detector's NATIVE EBNF
                # (bare-JSON json_schema degenerates live on multi-tool
                # required: 1-token EOS with 12 tools, CR loop with 2).
                self.assertEqual(res.tool_call_constraint[0], "ebnf")
                self.assertIs(
                    (req.chat_template_kwargs or {}).get("enable_thinking"),
                    False,
                    "forced tool_choice must disable thinking for from-token-0 enforcement",
                )

    def test_32_specific_function_forces_native_ebnf(self):
        svc = self._make_serving("qwen3", "qwen3_coder")
        req = self._make_request(
            {"type": "function", "function": {"name": "calculator"}}
        )
        res = svc._process_messages(req, is_multimodal=False)
        self.assertIsNotNone(res.tool_call_constraint)
        self.assertEqual(res.tool_call_constraint[0], "ebnf")
        self.assertIs(
            (req.chat_template_kwargs or {}).get("enable_thinking"), False
        )

    def test_33_forcing_grammar_rejects_free_text(self):
        try:
            import importlib.util

            if importlib.util.find_spec("llguidance") is None:
                self.skipTest("llguidance not installed")
        except (ImportError, ValueError):
            self.skipTest("llguidance not importable")
        import os

        from llguidance import LLTokenizer

        from sgl_jax.srt.constrained.base_grammar_backend import INVALID_GRAMMAR_OBJ
        from sgl_jax.srt.constrained.llguidance_backend import GuidanceBackend

        tok_path = os.environ.get("SGL_JAX_TEST_TOKENIZER_JSON")
        if not tok_path or not os.path.exists(tok_path):
            self.skipTest("SGL_JAX_TEST_TOKENIZER_JSON not set")
        svc = self._make_serving("qwen3", "qwen3_coder")
        req = self._make_request("required")
        res = svc._process_messages(req, is_multimodal=False)
        self.assertEqual(res.tool_call_constraint[0], "ebnf")
        ebnf = res.tool_call_constraint[1]
        self.assertIsInstance(ebnf, str)
        tok = LLTokenizer(tok_path)
        g = GuidanceBackend(tokenizer=tok).dispatch_ebnf(ebnf)
        self.assertIsNot(g, INVALID_GRAMMAR_OBJ)
        toks = tok.greedy_tokenize("7 times 8 is 56.")
        g.ll_matcher.consume_token(toks[0])
        self.assertTrue(
            g.ll_matcher.is_error(),
            "forcing native EBNF must reject free-text answer",
        )

    def test_34_native_ebnf_accepts_newlined_envelope(self):
        """The qwen3_coder EBNF must accept the model's native envelope with
        newlines (<tool_call>\\n<function=...>...), which the composed root
        rule previously rejected (strict "<tool_call>" adjacency)."""
        try:
            import importlib.util

            if importlib.util.find_spec("llguidance") is None:
                self.skipTest("llguidance not installed")
        except (ImportError, ValueError):
            self.skipTest("llguidance not importable")
        import os

        from llguidance import LLTokenizer

        from sgl_jax.srt.constrained.base_grammar_backend import INVALID_GRAMMAR_OBJ
        from sgl_jax.srt.constrained.llguidance_backend import GuidanceBackend
        from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser

        tok_path = os.environ.get("SGL_JAX_TEST_TOKENIZER_JSON")
        if not tok_path or not os.path.exists(tok_path):
            self.skipTest("SGL_JAX_TEST_TOKENIZER_JSON not set")
        calc = make_tool("calculator", {"expression": {"type": "string"}})
        p = FunctionCallParser([calc], "qwen3_coder")
        ebnf = p.get_ebnf("required")
        self.assertIsNotNone(ebnf)
        tok = LLTokenizer(tok_path)
        g = GuidanceBackend(tokenizer=tok).dispatch_ebnf(ebnf)
        self.assertIsNot(g, INVALID_GRAMMAR_OBJ)
        good = (
            "<tool_call>\n<function=calculator>\n<parameter=expression>\n"
            "7*8\n</parameter>\n</function>\n</tool_call>"
        )
        for i in tok.greedy_tokenize(good):
            g.ll_matcher.consume_token(i)
            if g.ll_matcher.is_error():
                self.fail(f"native EBNF rejected valid tool call: {g.ll_matcher.get_error()[:300]}")
        # And the serving-layer parser must extract the call from it.
        r = p.detector.detect_and_parse(good, [calc])
        self.assertTrue(r.calls)
        self.assertEqual(r.calls[0].name, "calculator")


if __name__ == "__main__":
    unittest.main()




