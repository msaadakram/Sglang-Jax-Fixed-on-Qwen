"""Regression test for the token-id-0 decode loop under grammar constraints.

Upstream reference: sgl-project/sglang#36537 (Qwen3.8 thinking + tools +
qwen3_coder loops on token id 0).

Root cause (JAX runtime): ``allocate_token_bitmask`` allocated a *zeroed*
bitmask. In llguidance's convention a set bit means "token allowed", so a row
that is allocated but never filled (grammar errored/finished mid-request, or
a request without a grammar batched next to constrained requests) masked out
the *entire* vocabulary. With every logit set to -inf, greedy sampling
deterministically selects token id 0, producing the infamous
"!!!!!!!!!!" repetition loop.

The mask contract tested here: a row that is allocated but not filled must
allow every token (unconstrained), never mask everything.

Run with:
    python -m pytest test/srt/constrained/test_bitmask_default_allow.py
"""

import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.constrained.bitmask_ops import allocate_token_bitmask, unpack_bitmask


def _all_allowed(mask_row_int32: np.ndarray, vocab_size: int) -> bool:
    unpacked = unpack_bitmask(jnp.asarray(mask_row_int32[None, ...]))
    return bool(jnp.all(unpacked[0, :vocab_size]))


class _FinishedGrammarStub:
    """A grammar whose matcher errored / finished mid-request.

    Mirrors GuidanceGrammar after an error: ``finished`` is True, so
    update_grammar_vocab_mask skips fill_vocab_mask for this row.
    """

    def __init__(self):
        self.finished = True
        self.fill_called = False

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int) -> np.ndarray:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: np.ndarray, idx: int):
        self.fill_called = True

    def is_terminated(self) -> bool:
        return True


class TestAllocateTokenBitmaskDefaultsToAllowAll(unittest.TestCase):
    VOCAB = 131  # not a multiple of 32, exercises the tail word

    def test_fresh_allocation_allows_all_tokens(self):
        mask = allocate_token_bitmask(2, self.VOCAB)
        self.assertEqual(mask.dtype, np.int32)
        for row in range(2):
            self.assertTrue(
                _all_allowed(mask[row], self.VOCAB),
                "a never-filled mask row must allow every token; a zeroed row "
                "masks the whole vocab and forces token id 0",
            )

    def test_unfilled_row_in_mixed_batch_allows_all(self):
        """Batch mixing a constrained request (row 1) with an unconstrained
        one (row 0, grammar=None): row 0 must stay unconstrained."""
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo

        grammar = _FinishedGrammarStub()
        info = ModelWorkerSamplingInfo(
            temperatures=np.array([0.7, 0.7], dtype=np.float32),
            top_ps=np.array([1.0, 1.0], dtype=np.float32),
            top_ks=np.array([-1, -1], dtype=np.int32),
            min_ps=np.array([0.0, 0.0], dtype=np.float32),
            vocab_size=self.VOCAB,
            grammars=[None, grammar],
        )
        info.update_grammar_vocab_mask()
        self.assertIsNotNone(info.vocab_mask)
        self.assertTrue(
            _all_allowed(info.vocab_mask[0], self.VOCAB),
            "unconstrained row in a mixed batch was fully masked (token-0 loop)",
        )

    def test_finished_grammar_row_allows_all(self):
        """After a grammar error mid-generation the row is never filled
        again; it must fall back to unconstrained decoding, not to a
        fully-masked row."""
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo

        grammar = _FinishedGrammarStub()
        info = ModelWorkerSamplingInfo(
            temperatures=np.array([0.7, 0.7], dtype=np.float32),
            top_ps=np.array([1.0, 1.0], dtype=np.float32),
            top_ks=np.array([-1, -1], dtype=np.int32),
            min_ps=np.array([0.0, 0.0], dtype=np.float32),
            vocab_size=self.VOCAB,
            grammars=[grammar, grammar],
        )
        info.update_grammar_vocab_mask()
        self.assertFalse(grammar.fill_called)
        for row in (0, 1):
            self.assertTrue(
                _all_allowed(info.vocab_mask[row], self.VOCAB),
                "finished/errored grammar row was fully masked (token-0 loop)",
            )


if __name__ == "__main__":
    unittest.main()
