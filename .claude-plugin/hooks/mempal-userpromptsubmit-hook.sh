#!/bin/bash
# MemPalace UserPromptSubmit Hook — thin wrapper calling Python CLI
# Fires whenever the user submits a prompt; when MEMPALACE_LOCAL_RETRIEVAL=1
# and there is an active intent, surfaces top-k relevant memories as
# additionalContext so the model sees them before it plans.
# All logic lives in mempalace.hooks_cli for cross-harness extensibility.
INPUT=$(cat)
echo "$INPUT" | python3 -m mempalace hook run --hook userpromptsubmit --harness claude-code
