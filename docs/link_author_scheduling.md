# link-author scheduling

The `mempalace link-author process` command drains the candidate queue
(filled in-session by `tool_finalize_intent`) by asking an
Opus-designed / Haiku-run jury to decide whether each pair should be
connected and with which predicate. Only LLM-accepted edges land in
the KG.

Two triggers, both OK to have enabled:

1. **Finalize-triggered detached subprocess** (primary, zero-config).
   Every successful `tool_finalize_intent` calls `_dispatch_if_due`
   which — after a 1-hour cadence gate and a pending-candidates check
   — spawns `mempalace link-author process` in a detached subprocess.
   Non-blocking. Inherits environment from the parent (so a
   `.env`-loaded `ANTHROPIC_API_KEY` flows through).

2. **Cron / launchd / Task Scheduler** (belt-and-suspenders). For
   environments where the MCP server is restarted often, or when you
   want a guaranteed periodic sweep independent of agent activity.

## 1. Create a dedicated mempalace API key

Go to <https://console.anthropic.com/settings/keys> and create a new
key. Name it `mempalace-link-author` for rotation clarity — keeping
it separate from your Flowserv / paperclip / any-other-project keys
means separate billing attribution and you can revoke just this key
without breaking other tools.

## 2. Drop the key in `<palace>/.env`

Copy the repo's `.env.example` to `<palace_path>/.env` and paste in
your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`<palace_path>` is whatever `mempalace status` reports as your palace
root (default: `~/.mempalace/palace`).

`.env` is gitignored. Never commit a real key anywhere.

## 3. Verify the key works

```
mempalace link-author process --dry-run
```

Expected behavior by exit code:

| Exit code | Meaning                                                    |
|-----------|------------------------------------------------------------|
| 0         | OK. Key valid, pipeline ran (dry-run = no writes).         |
| 2         | Missing / malformed / rejected key. Check `.env` contents. |
| 3         | Anthropic API unreachable (network / 5xx).                 |

If `--dry-run` exits 0, the scheduler will too.

## 4. Schedule a periodic run (optional)

### Linux / macOS — cron

The MCP-server `_dispatch_if_due` path is usually enough, but a cron
job is useful if you want a guaranteed periodic sweep.

Create a wrapper script so the `.env` gets loaded into the
environment before the CLI runs:

```bash
# ~/bin/mempalace-link-author.sh
#!/usr/bin/env bash
set -euo pipefail
PALACE="${MEMPALACE_PALACE_PATH:-$HOME/.mempalace/palace}"
set -a
source "$PALACE/.env"
set +a
exec mempalace link-author process >> "$PALACE/link_author.log" 2>&1
```

`chmod +x ~/bin/mempalace-link-author.sh`, then add to `crontab -e`:

```cron
# Every hour at :07
7 * * * * /home/you/bin/mempalace-link-author.sh
```

### macOS — launchd

Create `~/Library/LaunchAgents/com.mempalace.link-author.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.mempalace.link-author</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/bin/mempalace-link-author.sh</string>
  </array>
  <key>StartInterval</key>     <integer>3600</integer>
  <key>StandardOutPath</key>   <string>/Users/you/.mempalace/palace/link_author.log</string>
  <key>StandardErrorPath</key> <string>/Users/you/.mempalace/palace/link_author.log</string>
</dict>
</plist>
```

Load it: `launchctl load ~/Library/LaunchAgents/com.mempalace.link-author.plist`.

The wrapper script approach (same as cron) is the simplest; launchd's
`EnvironmentVariables` key can also inline the key, but that puts it
in a plist which ends up backed up with your macOS profile — keep it
in `.env` instead.

### Windows — Task Scheduler

Create `C:\Users\you\bin\mempalace-link-author.cmd`:

```bat
@echo off
set PALACE=%USERPROFILE%\.mempalace\palace
for /f "usebackq tokens=1* delims==" %%a in ("%PALACE%\.env") do set %%a=%%b
mempalace link-author process >> "%PALACE%\link_author.log" 2>&1
```

Then in Task Scheduler:

1. Create Task → General: name `mempalace-link-author`, "Run whether
   user is logged on or not".
2. Triggers: Daily, Repeat task every 1 hour for a duration of 1 day,
   Enabled.
3. Actions: Start a program, Program = the `.cmd` path above.
4. Conditions: uncheck "Start the task only if the computer is on AC
   power" if you're on a laptop.

## 5. Monitoring

Check recent runs:

```
mempalace link-author status --recent 10
```

Audit predicate creations (jury-proposed new predicates are logged
here for periodic human review — if the predicate space is ballooning
with synonyms, the near-duplicate threshold needs tuning):

```
mempalace link-author status --new-predicates
```

Raw SQLite:

```sql
SELECT * FROM link_prediction_candidates
  ORDER BY processed_ts DESC NULLS FIRST LIMIT 20;
SELECT * FROM link_author_runs ORDER BY started_ts DESC LIMIT 10;
```

## 6. Rotation

1. Revoke the old key at <https://console.anthropic.com/settings/keys>.
2. Generate a new one.
3. Replace the value in `<palace>/.env`.
4. `mempalace link-author process --dry-run` to confirm.

No MCP-server restart needed — the CLI reads `.env` on every
invocation, and the finalize-triggered subprocess inherits the MCP
server's environment only insofar as the server loaded `.env` at
startup. For the cleanest pickup, restart the MCP server once after
rotating (one claude-code session-close / session-open round-trip).

## 7. Cost & pacing

Per candidate in a non-batched run:

- 1 × Opus call (~1 KB input, ~1 KB output)
- 3 × Haiku calls in parallel (~2 KB input each)
- 1 × Haiku synthesis call (~1 KB input, ~500 B output)

With default batching (cosine ≥ 0.9 on domain-hint embeddings), Opus
calls compress roughly 2-4× on normal workloads. Haiku is cheap
enough that the jury execution dominates wall time but not cost.

`--max 50` (default) caps a run at 50 candidates. At the default
1-hour cadence that's a 50-candidate/hour ceiling; if you're seeing
the queue grow faster, drop `--max` for more-frequent-but-smaller
runs or bump the threshold so only higher-signal pairs make it in.
