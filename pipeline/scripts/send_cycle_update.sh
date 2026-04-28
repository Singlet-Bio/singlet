#!/bin/bash
# send_cycle_update.sh — DISABLED (2026-04-15). All emails now via hourly_email.py.
echo "[$(date '+%H:%M:%S')] send_cycle_update.sh: email sending disabled; skipping."
exit 0
# Original: send a development cycle update email to debruinz@gvsu.edu
#     --after "28.1s" \
#     --gain "13.3%" \
#     --next "S3: K-mer pre-screening" \
#     --body "Multi-line paragraph description goes here..." \
#     --bullets "* M2.1: 28.1s (was 32.4s, -13.3%)
# * M3.1 Pearson r: 0.9998 (no regression)
# * Next: S3 k-mer pre-screening"
#
# Subject line format: "<headline> [singlify-perf]"
# --headline should be 5-6 words capturing the key result.
#
# To steer the agent between cycles, edit:
#   singlify/scripts/USER_NOTES.md
# The agent reads this file at the start of every cycle.

set -euo pipefail

CYCLE=""
HEADLINE=""
HYPOTHESIS=""
RESULT=""
BEFORE=""
AFTER=""
GAIN=""
NEXT=""
BODY=""
BULLETS=""
TO="debruinz@gvsu.edu"
CHECKPOINT="/mnt/home/debruinz/Singlet-AI/singlify/scripts/cycle_history.tsv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cycle)       CYCLE="$2"; shift 2 ;;
        --headline)    HEADLINE="$2"; shift 2 ;;
        --hypothesis)  HYPOTHESIS="$2"; shift 2 ;;
        --result)      RESULT="$2"; shift 2 ;;
        --before)      BEFORE="$2"; shift 2 ;;
        --after)       AFTER="$2"; shift 2 ;;
        --gain)        GAIN="$2"; shift 2 ;;
        --next)        NEXT="$2"; shift 2 ;;
        --body)        BODY="$2"; shift 2 ;;
        --bullets)     BULLETS="$2"; shift 2 ;;
        --to)          TO="$2"; shift 2 ;;
        # Legacy / ignored flags
        --table-tsv)   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# If body not set, read from stdin if it's a pipe
if [[ -z "$BODY" ]] && [[ ! -t 0 ]]; then
    BODY=$(cat)
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M %Z')

# Append to cycle history TSV (kept for records; not emailed)
if [[ ! -f "$CHECKPOINT" ]]; then
    echo -e "cycle\ttimestamp\thypothesis\tresult\tbefore\tafter\tgain\tnext" > "$CHECKPOINT"
fi
if [[ -n "$RESULT" && "$RESULT" != "in_progress" ]]; then
    echo -e "${CYCLE}\t${TIMESTAMP}\t${HYPOTHESIS}\t${RESULT}\t${BEFORE}\t${AFTER}\t${GAIN}\t${NEXT}" >> "$CHECKPOINT"
fi

# Subject line: "<headline> [singlify-perf]"
# Fall back to a minimal subject if no headline provided.
if [[ -n "$HEADLINE" ]]; then
    SUBJECT="${HEADLINE} [singlify-perf]"
else
    SUBJECT="Cycle ${CYCLE} update [singlify-perf]"
fi

# Result label (plain text)
RESULT_LABEL=""
case "$RESULT" in
    win)      RESULT_LABEL="WIN" ;;
    dead_end) RESULT_LABEL="DEAD END" ;;
    *)        RESULT_LABEL="IN PROGRESS" ;;
esac

# Build bullet list: prefer --bullets arg, otherwise synthesise from params
if [[ -z "$BULLETS" ]]; then
    BULLETS="* Result: ${RESULT_LABEL}
* Cycle: ${CYCLE}
* Hypothesis: ${HYPOTHESIS}
* Before: ${BEFORE}
* After: ${AFTER}
* Gain: ${GAIN}
* Next: ${NEXT}"
fi

# Build plain-text email body — plain paragraphs, no markup, no ASCII art
if [[ -n "$RESULT_LABEL" && "$RESULT_LABEL" != "IN PROGRESS" ]]; then
    RESULT_SENTENCE="This cycle ended as a ${RESULT_LABEL}."
else
    RESULT_SENTENCE="Work is ongoing."
fi

TIMING_SENTENCE=""
if [[ -n "$BEFORE" && -n "$AFTER" && -n "$GAIN" ]]; then
    TIMING_SENTENCE=" Performance moved from ${BEFORE} to ${AFTER}, a change of ${GAIN}."
fi

NEXT_SENTENCE=""
if [[ -n "$NEXT" ]]; then
    NEXT_SENTENCE="Next up is ${NEXT}."
fi

PLAIN_BODY="Singlify development update — cycle ${CYCLE}
${TIMESTAMP}

${RESULT_SENTENCE}${TIMING_SENTENCE}

${BODY}

${NEXT_SENTENCE}

To steer the agent between cycles, edit singlify/scripts/USER_NOTES.md and the agent will confirm your instructions in its next email."

# Send plain-text email via Python smtplib (reliable, logs errors)
EMAIL_LOG="/mnt/home/debruinz/Singlet-AI/singlify/scripts/email_errors.log"

EMAIL_TO="$TO" EMAIL_SUBJECT="$SUBJECT" EMAIL_BODY="$PLAIN_BODY" \
python3 -c '
import smtplib, sys, socket, os
from email.message import EmailMessage

to_addr = os.environ["EMAIL_TO"]
subject = os.environ["EMAIL_SUBJECT"]
body = os.environ["EMAIL_BODY"]

msg = EmailMessage()
msg["From"] = f"singlify-agent@{socket.getfqdn()}"
msg["To"] = to_addr
msg["Subject"] = subject
msg.set_content(body)

try:
    with smtplib.SMTP("localhost", timeout=10) as s:
        s.send_message(msg)
    print(f"[OK] Email sent: \"{subject}\" -> {to_addr}")
except Exception as e:
    ts = __import__("datetime").datetime.now().isoformat()
    print(f"[{ts}] SMTP FAIL: {e}", file=sys.stderr)
    import subprocess
    try:
        proc = subprocess.run(["/usr/sbin/sendmail", "-t"],
            input=str(msg), text=True, capture_output=True, timeout=10)
        if proc.returncode == 0:
            print(f"[OK] Email sent via sendmail fallback: \"{subject}\" -> {to_addr}")
        else:
            print(f"[{ts}] sendmail FAIL (rc={proc.returncode}): {proc.stderr}", file=sys.stderr)
            sys.exit(1)
    except Exception as e2:
        print(f"[{ts}] sendmail fallback FAIL: {e2}", file=sys.stderr)
        sys.exit(1)
' 2>>"$EMAIL_LOG"

EMAIL_RC=$?
if [[ $EMAIL_RC -ne 0 ]]; then
    echo "[$(date '+%H:%M:%S')] [WARN] Email send FAILED (see $EMAIL_LOG); cycle data saved to $CHECKPOINT"
else
    echo "[$(date '+%H:%M:%S')] Email sent: '$SUBJECT' → $TO"
fi
