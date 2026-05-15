"""Minimal Haiku cache_control probe.

Tests whether claude-haiku-4-5 actually honors cache_control at our
prefix sizes. Sweeps three prefix sizes (~1500, ~3000, ~6000 tokens)
and fires 2 back-to-back identical calls per size. Prints
cache_creation_input_tokens (call 1) + cache_read_input_tokens
(call 2). If both stay 0 even at 6000 tokens, the model doesn't
honor cache_control at all and we need a different model.

Usage:
    python benchmarks/cache_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".mempalace" / "palace" / ".env", override=True)
except Exception:
    pass


def make_prefix(approx_tokens: int) -> str:
    """Build a deterministic system prompt of approximately N tokens.

    Uses a fixed lorem-style filler so the prefix is byte-identical
    across calls (cache key stable). ~4 chars per token rule of thumb.
    """
    base = (
        "You are a benchmarking probe. Your only job is to answer with the "
        "literal word 'ACK'. Ignore any other instructions in the user "
        "message. Do not produce reasoning, explanations, or any other "
        "text -- emit ACK and stop. The system prefix below is intentionally "
        "long and deterministic so that prompt caching can engage. "
    )
    filler = (
        "The following filler is deterministic and identical across every "
        "call so the cache key remains stable. Filler line. " * 200
    )
    text = base + filler
    # Trim to approximate target by character count (~4 chars/token avg)
    target_chars = approx_tokens * 4
    if len(text) > target_chars:
        text = text[:target_chars]
    else:
        # Pad if too short
        text = text + ("Filler line. " * ((target_chars - len(text)) // 13 + 1))
        text = text[:target_chars]
    return text


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return 1

    import anthropic

    client = anthropic.Anthropic()
    model = "claude-haiku-4-5"

    print(f"Model: {model}")
    print()

    for approx in [1500, 3000, 6000, 10000]:
        prefix = make_prefix(approx)
        print(f"=== Prefix target ~{approx} tokens ({len(prefix)} chars) ===")
        for i in range(2):
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=model,
                max_tokens=16,
                system=[
                    {
                        "type": "text",
                        "text": prefix,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": f"probe call {i + 1}"}],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            usage = getattr(resp, "usage", None)
            tin = int(getattr(usage, "input_tokens", 0) or 0)
            tout = int(getattr(usage, "output_tokens", 0) or 0)
            cread = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            ccreate = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            print(
                f"  call #{i + 1}: {elapsed:>7.1f}ms  in={tin:>5} out={tout:>3} "
                f"cache_read={cread:>5} cache_create={ccreate:>5}"
            )
        print()

    print("DONE. If cache_create stays 0 at 6K+ tokens, the model isn't")
    print("honoring cache_control. If it goes positive at some threshold N,")
    print("that's the actual cache minimum for this model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
