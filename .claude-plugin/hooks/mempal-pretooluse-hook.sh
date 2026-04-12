#!/bin/bash
# MemPalace PreToolUse Hook — intent-based tool permission enforcement
# Reads active intent from ~/.mempalace/hook_state/active_intent_{session_id}.json
# All logic lives in mempalace.hooks_cli for cross-harness extensibility
INPUT=$(cat)
echo "$INPUT" | python3 -m mempalace hook run --hook pretooluse --harness claude-code
