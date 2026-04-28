#!/usr/bin/env bash
# send_status_email.sh — DISABLED (2026-04-15). All emails now via hourly_email.py.
# Keeping steering.txt check for backward compat with agents that call this script.
echo "[$(date '+%H:%M:%S')] send_status_email.sh: email sending disabled; skipping."
exit 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TO="debruinz@gvsu.edu"

# --- 1. Check steering.txt ---
STEERING="$SCRIPT_DIR/steering.txt"
STEERING_MSG=""
if [[ -f "$STEERING" ]] && grep -qv '^#' "$STEERING" 2>/dev/null && [[ -s "$STEERING" ]]; then
    # Has non-comment, non-empty content
    STEERING_MSG=$(grep -v '^#' "$STEERING" | sed '/^[[:space:]]*$/d')
fi

# --- 2. Check IMAP inbox for steering replies ---
if [[ -z "$STEERING_MSG" ]]; then
    IMAP_MSG=$(python3 "$SCRIPT_DIR/poll_steering_email.py" --clear 2>/dev/null || true)
    if [[ -n "$IMAP_MSG" ]]; then
        STEERING_MSG="$IMAP_MSG"
    fi
fi

# --- 3. Clear steering.txt if it had content ---
if [[ -n "$STEERING_MSG" ]] && grep -qv '^#' "$STEERING" 2>/dev/null; then
    # Keep only comment lines
    grep '^#' "$STEERING" > "${STEERING}.tmp" && mv "${STEERING}.tmp" "$STEERING" || true
fi

# --- 4. Generate and send HTML status email ---
python3 "$SCRIPT_DIR/generate_status.py" --email | \
# s-nail interprets -a as file attachments, not custom headers.
# Use sendmail directly for HTML email with proper MIME headers.
{
    echo "To: ${TO}"
    echo "Subject: singlify status — $(date '+%Y-%m-%d %H:%M')"
    echo "Reply-To: ${TO}"
    echo "MIME-Version: 1.0"
    echo "Content-Type: text/html; charset=UTF-8"
    echo ""
    python3 "$SCRIPT_DIR/generate_status.py" --email
} | /usr/sbin/sendmail -t \
    && echo "[$(date '+%H:%M:%S')] Status email sent → $TO" \
    || echo "[$(date '+%H:%M:%S')] WARNING: status email failed (sendmail error)"

# --- 5. Print steering prompt to stdout for agent to incorporate ---
if [[ -n "$STEERING_MSG" ]]; then
    echo ""
    echo "STEERING_PROMPT_FOR_AGENT:"
    echo "$STEERING_MSG"
fi

