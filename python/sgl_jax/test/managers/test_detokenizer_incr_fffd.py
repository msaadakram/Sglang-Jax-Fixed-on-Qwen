"""Regression tests: incremental detokenization must not emit U+FFFD for
multi-byte characters split across tokens (Qwen3.8 box-drawing corners).

Root cause: Qwen3.8 (vocab 248044) splits box-drawing corner characters into
two tokens whose UTF-8 bytes straddle the token boundary, e.g.

    \u250c = [11287, 234]   (E2 | 94 8C)
    \u2510 = [11287, 238]   (E2 | 94 90)
    \u2514 = [11287, 242]   (E2 | 94 94)
    \u2518 = [11287, 246]   (E2 | 94 98)

Decoding the first token of such a pair on its own yields U+FFFD. The old
detokenizer emitted that U+FFFD as a streaming chunk and then swallowed the
completed character when the second token arrived, so TUI frames streamed to
clients with corrupted corners. Line characters (\u2500, \u2502) are single
tokens, which is why only corners were affected.

The fix (backported from the CUDA SGLang incremental-detokenizer change):
  * never emit streaming text that ends in U+FFFD;
  * emit only the printable prefix (find_printable_text) in that case and do
    NOT advance the token offsets, so the next step re-decodes with the
    continuation token;
  * commit clean text and remember how much was already sent (sent_offset vs
    decoded_text_len) so the completed character is emitted exactly once.

These tests drive the real DetokenizerManager.handle_batch_token_id_out with
scheduler-faithful batches (5-token surrounding context, decode_ids sliced to
new tokens only via send_decode_id_offset semantics) and a byte-level fake
tokenizer that reproduces the verified Qwen3.8-27B decode behavior for the
tokens above, so they run deterministically without network access.
"""

import dataclasses

import pytest

from sgl_jax.srt.managers.detokenizer_manager import (
    DecodeStatus,
    DetokenizerManager,
    LimitedCapacityDict,
)
from sgl_jax.srt.managers.io_struct import BatchTokenIDOut
from sgl_jax.utils import find_printable_text

FFFD = "\ufffd"

# Token IDs -> raw UTF-8 bytes, reproducing the verified Qwen3.8-27B
# (byte-level BPE) decode behavior for the tokens involved.
_TOKEN_BYTES = {
    11287: b"\xe2",        # first byte of a corner char; decodes alone to U+FFFD
    234: b"\x94\x8c",      # completes \u250c
    238: b"\x94\x90",      # completes \u2510
    242: b"\x94\x94",      # completes \u2514
    246: b"\x94\x98",      # completes \u2518
    14136: b"\xe2\x94\x80",  # \u2500 (single token)
    70410: b"\xe2\x94\x82",  # \u2502 (single token)
    10: b"\n",
    32: b" ",
    1234: b"status",
    5678: b":",
}


class FakeQwen38Tokenizer:
    """Byte-level decoder stand-in for the Qwen3.8-27B tokenizer."""

    all_special_ids = []

    def _decode_ids(self, ids, **_kwargs):
        return b"".join(_TOKEN_BYTES[i] for i in ids).decode("utf-8", "replace")

    def decode(self, ids, **kwargs):
        return self._decode_ids(ids, **kwargs)

    def batch_decode(self, id_lists, **kwargs):
        return [self._decode_ids(ids, **kwargs) for ids in id_lists]


# Sanity: the fake reproduces the verified real-tokenizer behavior.
def test_fake_tokenizer_matches_verified_qwen38_behavior():
    tok = FakeQwen38Tokenizer()
    assert tok.decode([11287]) == FFFD
    assert tok.decode([11287, 234]) == "\u250c"
    assert tok.decode([11287, 238]) == "\u2510"
    assert tok.decode([11287, 242]) == "\u2514"
    assert tok.decode([11287, 246]) == "\u2518"
    assert tok.decode([14136]) == "\u2500"
    assert tok.decode([70410]) == "\u2502"


def _make_manager():
    # Bypass __init__ (zmq sockets, server args); only the attributes used by
    # handle_batch_token_id_out are needed.
    mgr = DetokenizerManager.__new__(DetokenizerManager)
    mgr.tokenizer = FakeQwen38Tokenizer()
    mgr.decode_status = LimitedCapacityDict(1 << 16)
    mgr.disable_tokenizer_batch_decode = False
    mgr.is_tool_call_parser_gpt_oss = False
    return mgr


@dataclasses.dataclass
class _Recv:
    """Minimal BatchTokenIDOut for a single request."""

    decode_ids: list
    finished: bool = False


def _build_recv(decode_ids, finished=False):
    fin = {"type": "length", "length": 1, "text": ""} if finished else None
    return BatchTokenIDOut(
        rids=["regress-rid-0"],
        finished_reasons=[fin],
        decoded_texts=[""],
        decode_ids=[decode_ids],
        read_offsets=[5],
        output_ids=None,
        skip_special_tokens=[False],
        spaces_between_special_tokens=[False],
        no_stop_trim=[False],
        prompt_tokens=[10],
        completion_tokens=[1],
        cached_tokens=[0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=None,
        output_hidden_states_for_mm=None,
    )


# 5 prompt tokens of surrounding context (INIT_INCREMENTAL_DETOKENIZATION_OFFSET).
_SURR = [1234, 5678, 32, 1234, 5678]


def _stream_one_token_at_a_time(mgr, out_ids):
    """Feed out_ids one token per batch (stream_interval=1), scheduler-faithful:
    first batch carries surr+token, later batches carry only the new token."""
    chunks = []
    first = True
    for k, tid in enumerate(out_ids):
        ids = (_SURR + [tid]) if first else [tid]
        first = False
        out = mgr.handle_batch_token_id_out(_build_recv(ids, finished=(k == len(out_ids) - 1)))
        chunks.append(out.output_strs[0])
    return chunks


def test_find_printable_text():
    assert find_printable_text("") == ""
    assert find_printable_text("abc") == ""  # no trailing space -> nothing safe
    assert find_printable_text("abc def") == "abc "
    assert find_printable_text("abc def ") == "abc def "
    assert find_printable_text("x\n") == "x\n"  # newline flushes
    assert find_printable_text("\u4f60") == "\u4f60"  # CJK last char
    assert find_printable_text("a\u4f60b") == "a\u4f60"  # CJK penultimate


def test_no_fffd_and_exact_text_for_split_corners():
    """The core regression: stream a TUI frame containing all four corner
    characters (each split across two tokens), one token at a time.
    No chunk may contain U+FFFD and the concatenated output must be exact."""
    mgr = _make_manager()
    out_ids = [
        11287, 234,  # \u250c
        14136,       # \u2500
        11287, 238,  # \u2510
        10,          # \n
        70410, 32, 1234, 5678, 32, 70410,  # | status: |
        10,          # \n
        11287, 242,  # \u2514
        14136,       # \u2500
        11287, 246,  # \u2518
    ]
    expected = "\u250c\u2500\u2510\n\u2502 status: \u2502\n\u2514\u2500\u2518"
    chunks = _stream_one_token_at_a_time(mgr, out_ids)

    assert all(FFFD not in c for c in chunks), f"U+FFFD leaked into stream: {chunks!r}"
    joined = "".join(chunks)
    assert joined == expected
    # Every corner must appear exactly once (old code swallowed them).
    for corner in "\u250c\u2510\u2514\u2518":
        assert joined.count(corner) == 1


def test_corner_emitted_when_completion_token_arrives():
    """First half of a corner pair emits no U+FFFD; the completed character
    appears in (or after) the chunk for the second token."""
    mgr = _make_manager()
    chunks = _stream_one_token_at_a_time(mgr, [11287, 234])
    assert FFFD not in "".join(chunks)
    assert "".join(chunks) == "\u250c"
    assert "\u250c" not in chunks[0]  # not emitted before the pair completes
    assert "\u250c" in chunks[1]


def test_incomplete_step_does_not_advance_offsets():
    """After a step whose text ends in U+FFFD, the token offsets must be
    unchanged so the next step re-decodes the full pair (no swallowed char)."""
    mgr = _make_manager()
    out = mgr.handle_batch_token_id_out(_build_recv(_SURR + [11287]))
    s = mgr.decode_status["regress-rid-0"]
    assert out.output_strs[0] == ""
    assert s.surr_offset == 0 and s.read_offset == 5  # not advanced
    # Completion step decodes the whole pair and emits the corner once.
    out2 = mgr.handle_batch_token_id_out(_build_recv([234], finished=True))
    assert out2.output_strs[0] == "\u250c"
    assert FFFD not in out2.output_strs[0]


def test_printable_prefix_emitted_then_skipped_on_completion():
    """'status ' + first corner byte: the printable prefix 'status ' is emitted
    once (not the U+FFFD), and must NOT be re-sent when the corner completes."""
    mgr = _make_manager()
    out_ids = [1234, 32, 11287, 234]  # status + ' ' + \u250c
    chunks = _stream_one_token_at_a_time(mgr, out_ids)
    assert all(FFFD not in c for c in chunks)
    joined = "".join(chunks)
    assert joined == "status \u250c"
    assert joined.count("status ") == 1  # prefix not double-sent


def test_non_streaming_single_finished_batch():
    """All-at-once (non-streaming) path: finished in the first batch."""
    mgr = _make_manager()
    out_ids = [11287, 234, 14136, 11287, 238, 10, 70410, 11287, 242, 14136, 11287, 246]
    expected = "\u250c\u2500\u2510\n\u2502\u2514\u2500\u2518"
    out = mgr.handle_batch_token_id_out(_build_recv(_SURR + out_ids, finished=True))
    assert out.output_strs[0] == expected
    assert FFFD not in out.output_strs[0]


def test_decode_status_chunked_text_accounting():
    """DecodeStatus chunked appends must keep decoded_text_len consistent."""
    s = DecodeStatus(decoded_text="ab", decode_ids=[1, 2], surr_offset=0, read_offset=2)
    assert s.decoded_text_len == 2
    s.append_decoded_text("cd")
    s.append_decoded_text("")  # no-op
    assert s.decoded_text_len == 4
    assert s.get_decoded_text() == "abcd"
    s.append_decoded_text("ef")
    assert s.get_decoded_text() == "abcdef"
    assert s.decoded_text_len == 6
