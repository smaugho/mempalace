#!/bin/bash
# MemPalace SessionStart Hook -- thin wrapper calling Python CLI
# Fires on startup/resume/clear/compact; rehydrates active intent on compact/resume.
# All logic lives in mempalace.hooks_cli for cross-harness extensibility.
INPUT=$(cat)
echo "$INPUT" | python3 -m mempalace hook run --hook sessionstart --harness claude-code
