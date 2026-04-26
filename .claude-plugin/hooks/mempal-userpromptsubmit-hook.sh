#!/bin/bash
# MemPalace UserPromptSubmit Hook — thin wrapper calling Python CLI
# Fires whenever the user submits a prompt. Slice B-2 behaviour: persists
# the prompt into the per-session pending_user_messages queue and surfaces
# the pending ids + a pointer to mempalace_declare_user_intents as
# additionalContext, so the agent's first action covers every pending
# message before any other tool call. PreToolUse blocks all non-allowed
# tools until the queue is drained.
# Opt-out: MEMPALACE_USER_INTENT_DISABLED=1 or MEMPALACE_DISABLE_LOCAL_RETRIEVAL=1
# All logic lives in mempalace.hooks_cli for cross-harness extensibility.
INPUT=$(cat)
echo "$INPUT" | python3 -m mempalace hook run --hook userpromptsubmit --harness claude-code
