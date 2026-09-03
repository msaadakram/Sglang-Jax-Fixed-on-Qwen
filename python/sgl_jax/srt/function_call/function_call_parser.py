import logging
from typing import Any, Literal

from sgl_jax.srt.entrypoints.openai.protocol import (
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolChoice,
)
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import ToolCallItem
from sgl_jax.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sgl_jax.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sgl_jax.srt.function_call.mimo_detector import MiMoDetector
from sgl_jax.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sgl_jax.srt.function_call.qwen25_detector import Qwen25Detector
from sgl_jax.srt.function_call.ebnf_composer import EBNFComposer
from sgl_jax.srt.function_call.utils import get_json_schema_constraint

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: dict[str, type[BaseFormatDetector]] = {
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "mimo": MiMoDetector,
        "glm47": Glm47MoeDetector,
        "glm45": Glm4MoeDetector,
    }

    def __init__(self, tools: list[Tool], tool_call_parser: str):
        detector: type[BaseFormatDetector] = None
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            detector = detector_class()
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        if not self.tools:
            return full_text, []
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.
        """
        tool_structures: list[StructuresResponseFormat] = list()
        tool_trigger_set: set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            schema = function.parameters if function.strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema,  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )

    def get_structure_constraint(
        self,
        tool_choice: ToolChoice | Literal["auto", "required"],
        thinking: bool = False,
    ) -> tuple[str, Any] | None:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request
            thinking: Whether reasoning/thinking is enabled for this request.
                When True, a from-token-0 grammar masks the <think> token the
                model wants to emit first; at temperature 0 the model then
                deterministically emits the EOS/stop token instead
                (completion_tokens=1, finish_reason=stop, matched_stop=<stop id>).
                For forced tool_choice we therefore compose the detector's
                native EBNF with an optional <think>...</think> free-text
                prefix so thinking and the tool grammar no longer conflict.

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        # NOTE: structural_tag only supports JSON-compatible content between the begin and end.
        # It cannot parse or validate function call Pythonic or XML-ish syntax.
        if (
            self.detector.supports_structural_tag()
            and tool_choice == "auto"
            and any(tool.function.strict for tool in self.tools)
        ):
            strict_tag = self.get_structure_tag()
            return ("structural_tag", strict_tag)
        elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
            if thinking:
                tag = self._build_thinking_structural_tag(tool_choice)
                if tag is not None:
                    return ("structural_tag", tag)
                logger.warning(
                    "Detector does not support thinking-tolerant structural "
                    "tags; falling back to json_schema for forced tool_choice "
                    "(thinking conflicts with from-token-0 grammars)."
                )
            json_schema = get_json_schema_constraint(self.tools, tool_choice)
            if json_schema is None:
                return None
            return ("json_schema", json_schema)

    def _filtered_tools(self, tool_choice: ToolChoice | Literal["required"]):
        if isinstance(tool_choice, ToolChoice):
            return [t for t in self.tools if t.function.name == tool_choice.function.name]
        return self.tools

    def _build_thinking_structural_tag(self, tool_choice) -> Any | None:
        """Build a thinking-tolerant structural tag for forced tool_choice.

        A from-token-0 grammar (json_schema) masks the <think> token a
        thinking model emits first, which at temperature 0 collapses the
        generation to a single EOS/stop token. A structural tag instead lets
        llguidance run unconstrained free text (the reasoning) until the
        <tool_call> trigger appears, then applies the detector's native
        tool-call structure as a Lark grammar. Requires the detector's
        call format to be expressible as Lark (Qwen3-Coder XML format is).
        """
        from llguidance import StructTag
        from llguidance.gbnf_to_lark import any_to_lark

        if not getattr(self.detector, "supports_lark_structural_tag", False):
            return None
        filtered = self._filtered_tools(tool_choice)
        if not filtered:
            return None
        # Body of one tool call, without the <tool_call>/</tool_call>
        # wrapper: the structural tag's begin/end supply it, and llguidance
        # re-triggers on <tool_call> for parallel calls.
        body_ebnf = EBNFComposer.build_ebnf(
            filtered,
            function_format="xml",
            individual_call_start_token=None,
            individual_call_end_token=None,
            tool_call_separator=None,
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
            key_value_separator='"\\n"',
        )
        try:
            body_lark = any_to_lark(body_ebnf)
        except Exception as e:
            logger.warning("Failed to convert tool-call EBNF to Lark: %s", e)
            return None
        # Strip the per-grammar %llguidance header: nested grammars in a
        # multi-grammar definition carry no header (mirrors StructTag.to_grammar).
        body_lark = "\n".join(
            line for line in body_lark.splitlines() if not line.startswith("%llguidance")
        )

        # Main grammar: a lazy free-text lexeme (the model's reasoning
        # preamble, which may contain any tokens incl. special  thinking
        # sequences) that STOPS at the <tool_call> trigger (mirrors
        # StructTag.to_grammar's [lazy] trigger rule), then one or more
        # <tool_call> structures via the tag_body sub-grammar.
        #
        # NOTE:  thinking/ response must NOT be referenced as bare names --
        # llguidance resolves unquoted identifiers as special-token
        # references, and LLTokenizer rejects unknown ones ("unknown special
        # token"), which turns the constraint into INVALID_GRAMMAR_OBJ (no
        # constraint at all -> the model chats freely instead of forcing a
        # tool call). The lazy TAG_TEXT lexeme is the tokenizer-agnostic way
        # to allow free text up to the trigger.
        main_lark = (
            "%llguidance {}\n\n"
            "TAG_TEXT: /(.|\\n)*/\n"
            "start: tool_tag+\n"
            'tool_tag_trig[lazy]: TAG_TEXT "<tool_call>"\n'
            "tool_tag: tool_tag_trig /[ \\n\t]/* @tag_body /[ \\n\t]/* "
            f'"{self.detector.tool_call_end_token}"\n'
        )
        from sgl_jax.srt.entrypoints.openai.protocol import (
            StructuralTagResponseFormat,
        )

        # A "structural_tag" value whose "grammars" key carries the full
        # llguidance multi-grammar definition (main + tool-call body).
        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=[],
            triggers=[self.detector.tool_call_start_token],
            lark_grammars={
                "struct_tag": main_lark,
                "tag_body": body_lark,
            },
        )

    def get_ebnf(self, tool_choice: ToolChoice | Literal["required"]) -> str | None:
        """
        Get the EBNF grammar for the specified tool choice.

        Args:
            tool_choice: The tool choice specification

        Returns:
            EBNF grammar string, or None if no valid tools found

        Note:
            If a specific function is requested but not found in available tools,
            logs a warning and falls back to using all available tools for backward compatibility.
        """
        filtered_tools = []
        if isinstance(tool_choice, ToolChoice):
            fn_name = tool_choice.function.name
            filtered_tools = [t for t in self.tools if t.function.name == fn_name]

            # Check if the requested function exists in available tools
            if not filtered_tools:
                available_functions = [t.function.name for t in self.tools]
                logger.warning(
                    "Function '%s' not found in available tools. "
                    "Available functions: %s. "
                    "Skipping tool choice.",
                    fn_name,
                    available_functions,
                )

                # TODO: Return a 400 error instead of warning when adapter supports proper error handling
                # For now, fall back to return None
                return None
        else:
            filtered_tools = self.tools

        return self.detector.build_ebnf(filtered_tools)
