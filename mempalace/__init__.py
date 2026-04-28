"""MemPalace -- Give your AI a memory. No API key required."""

import logging
import os
import platform

from .cli import main  # noqa: E402
from .version import __version__  # noqa: E402

# ChromaDB 0.6.x ships a Posthog telemetry client whose capture() signature is
# incompatible with the bundled posthog library, producing noisy stderr warnings
# on every client operation ("Failed to send telemetry event … capture() takes
# 1 positional argument but 3 were given").  Silence just that logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Diagnostic sink for finalize-coverage + UserPromptSubmit paths.
# ON BY DEFAULT so regressions of the 2026-04-24 "coverage = 0.0 despite
# matching feedback" class of bug are caught on first occurrence rather
# than rediscovered after hours of puzzled retries. Set
# MEMPALACE_DISABLE_FINALIZE_DEBUG=1 to silence if verbose output is a
# problem in production; the diagnostic class of bug it catches tends
# to be invisible without the log, so keeping it live is worth the
# minor I/O cost (one line per finalize).
#
# Log path: ~/.mempalace/finalize_debug.log (WARNING level only -- no
# INFO spam). Rotates by nothing today; call it from cron or let it
# grow -- each line is ~200 bytes and finalize is rare enough that
# unbounded growth isn't a practical problem.
if not os.environ.get("MEMPALACE_DISABLE_FINALIZE_DEBUG"):
    _dbg_path = os.path.expanduser("~/.mempalace/finalize_debug.log")
    try:
        os.makedirs(os.path.dirname(_dbg_path), exist_ok=True)
        _dbg_handler = logging.FileHandler(_dbg_path, encoding="utf-8")
        _dbg_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        _dbg_handler.setLevel(logging.WARNING)
        for _logger_name in ("mempalace.finalize_debug", "mempalace.userprompt_debug"):
            _lg = logging.getLogger(_logger_name)
            _lg.setLevel(logging.WARNING)
            _lg.addHandler(_dbg_handler)
            _lg.propagate = False
    except Exception:
        pass  # Diagnostics must not break startup.

# ONNX Runtime's CoreML provider segfaults during vector queries on Apple Silicon.
# Force CPU execution unless the user has explicitly set a preference.
if platform.machine() == "arm64" and platform.system() == "Darwin":
    os.environ.setdefault("ORT_DISABLE_COREML", "1")

__all__ = ["main", "__version__"]
