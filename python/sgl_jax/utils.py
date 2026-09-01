import json
import logging
import traceback
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TypeBasedDispatcher:
    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


def _is_chinese_char(cp: int) -> bool:
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def find_printable_text(text: str) -> str:
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99
    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def get_exception_traceback() -> str:
    """Get the current exception traceback as a string."""
    return traceback.format_exc()


def convert_json_schema_to_str(json_schema: dict | str | type[BaseModel]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema, sort_keys=True)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def _create_dummy_buffer(buffer):
    """Create dummy buffer with sequential values, preserving type and sharding."""
    if hasattr(buffer, "value"):
        # It's a Param-wrapped value
        arr = buffer.value
        # Get sharding from the actual array, not the Param wrapper
        sharding = arr.sharding if hasattr(arr, "sharding") else None
        new_arr = jax.device_put(
            jnp.arange(arr.size, dtype=arr.dtype).reshape(arr.shape),
            device=sharding,
        )
        # Re-wrap in the same type (e.g., nnx.Param)
        return type(buffer)(value=new_arr)
    else:
        # It's a raw Array
        sharding = buffer.sharding if hasattr(buffer, "sharding") else None
        new_arr = jax.device_put(
            jnp.arange(buffer.size, dtype=buffer.dtype).reshape(buffer.shape),
            device=sharding,
        )
        return new_arr


def traverse_and_update(state_obj, target_modules):
    """
    Recursively traverse state structure and update A_buffer/B_buffer in target modules.

    Args:
        state_obj: Can be State/Params (dict-like), list, or leaf values (Param, Array, etc.)

    Returns:
        Updated state with same type as input
    """
    if target_modules is None or len(target_modules) == 0:
        return state_obj
    # Case 1: State or Params (dict-like with .items() method, but not a Param leaf node)
    if hasattr(state_obj, "items") and not hasattr(state_obj, "value"):
        updated = {}

        for key, value in state_obj.items():
            if key in ("A_buffer", "B_buffer"):
                # Found a LoRA buffer to replace
                updated[key] = _create_dummy_buffer(value)
            else:
                # Regular key, recurse normally
                updated[key] = traverse_and_update(value, target_modules)

        # Preserve type: return State if input was State, otherwise return same type
        return type(state_obj)(updated)

    # Case 2: List (e.g., layers list)
    elif isinstance(state_obj, list):
        return [traverse_and_update(item, target_modules) for item in state_obj]

    # Case 3: Leaf nodes (Param objects, raw arrays, or other values)
    else:
        return state_obj
