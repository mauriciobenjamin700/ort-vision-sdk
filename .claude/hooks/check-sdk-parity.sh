#!/usr/bin/env bash
# Warn when a commit touches only one of the two SDKs in this monorepo.
#
# ort-vision-sdk ships a Python package and an npm package that mirror each
# other's surface. A behaviour change landing in one alone is how the two drift:
# the Python side gained `normalization="auto"` while the web `Classifier` kept
# the ImageNet default, and `OrtSession.providers` was reconciled in Python while
# the TypeScript one still reported the list that had been requested.
#
# This warns; it never blocks. Plenty of commits are legitimately one-sided
# (build-time fusion is Python-only, canvas handling is web-only) — the point is
# that the asymmetry gets stated rather than assumed.
#
# No jq: this machine has none, and a hook that depends on a tool the developer
# may not have installed is a hook that silently stops working. The payload is
# matched as text, which is why an unrelated Bash call whose text mentions
# "git commit" can also trigger it. That costs one extra warning and nothing else.
#
# Running before the command means the index this reads is the index as it stands
# *now*. A single compound command that stages and commits together
# (`git add X && git commit ...`) is therefore judged on what was staged before
# its own `git add` — stage in one call and commit in the next to get a verdict
# on what is actually going in.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

payload="$(cat)"
printf '%s' "$payload" | grep -q 'git commit' || exit 0

# Only speak for this repository, never for another checkout the session has open.
toplevel="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
[ "$toplevel" = "$REPO" ] || exit 0

files="$(git -C "$REPO" diff --cached --name-only 2>/dev/null)"
if [ -z "$files" ]; then
  files="$(git -C "$REPO" diff --name-only HEAD 2>/dev/null)"
fi
[ -n "$files" ] || exit 0

touches_python=0
touches_web=0
printf '%s\n' "$files" | grep -q '^sdk-python/' && touches_python=1
printf '%s\n' "$files" | grep -q '^sdk-js-web/' && touches_web=1

if [ "$touches_python" -eq "$touches_web" ]; then
  exit 0
fi

if [ "$touches_python" -eq 1 ]; then
  lands="sdk-python/"
  missing="sdk-js-web/"
else
  lands="sdk-js-web/"
  missing="sdk-python/"
fi

cat <<JSON
{
  "systemMessage": "Paridade de SDK: este commit toca ${lands} e nao ${missing}.",
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": "Este commit toca ${lands} e nao ${missing}. Os dois SDKs espelham a mesma superficie publica, entao uma mudanca de comportamento que entra so de um lado e como eles divergem. Antes de commitar: verifique se o outro SDK precisa da mesma mudanca e faca-a, ou registre explicitamente no corpo do commit por que ela e de um lado so (fusao e build-time, so Python; canvas e so web; docs compartilhadas nao contam). Aviso apenas - nao bloqueia."
  }
}
JSON
