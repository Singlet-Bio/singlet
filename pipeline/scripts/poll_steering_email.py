#!/usr/bin/env python3
"""
poll_steering_email.py — poll Gmail INBOX for reply emails from the user.

Any reply to an agent email is treated as steering context. No special subject
tag needed. Staleness: if the email sat in the inbox for >6h before being read,
it is flagged as possibly referencing an earlier cycle.

Setup (one time):
  Create singlify/scripts/.env with:
    IMAP_HOST=imap.gmail.com
    IMAP_USER=debruinz@gvsu.edu
    IMAP_PASS=<google-app-password>   # myaccount.google.com/apppasswords

  The outbound agent emails must include a Reply-To: debruinz@gvsu.edu header
  so that user replies land back in the Gmail inbox (handled by send_*.sh).

Usage:
  python3 poll_steering_email.py          # print any pending steering message
  python3 poll_steering_email.py --clear  # mark messages as seen after reading

Exit codes:
  0  — steering message found (printed to stdout)
  1  — no message or .env absent (silent)
  2  — IMAP error (message printed to stderr)
"""
import imaplib
import email
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE    = os.path.join(SCRIPTS_DIR, ".env")

# Age threshold: replies older than this are flagged as possibly stale
STALE_HOURS = 6


def load_env():
    if not os.path.exists(ENV_FILE):
        return None
    cfg = {}
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def decode_str(s):
    if s is None:
        return ""
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    parts = decode_header(s)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(part)
    return "".join(out)


def get_text_body(msg):
    """Extract plain-text body. Fall back to HTML-stripped if no plain part."""
    plain, html_body = None, None
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            if ct == "text/plain" and plain is None:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                plain = payload.decode(charset, errors="replace")
            elif ct == "text/html" and html_body is None:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                html_body = payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_body = text
            else:
                plain = text

    if plain:
        return plain
    if html_body:
        # Strip HTML tags for a rough plain-text version
        html_body = re.sub(r"<br\s*/?>", "\n", html_body, flags=re.IGNORECASE)
        html_body = re.sub(r"<[^>]+>", "", html_body)
        return html_body
    return ""


def strip_reply_quoting(body):
    """Remove email reply quoting (lines starting with >, 'On ... wrote:', etc.)."""
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        # Stop at typical quoting markers
        if re.match(r"^On .{10,} wrote:$", stripped):
            break
        if re.match(r"^From:.*singlify", stripped, re.IGNORECASE):
            break
        if stripped.startswith(">"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_stale(msg):
    """Return True if this email has been sitting in the inbox for >STALE_HOURS."""
    date_str = msg.get("Date", "")
    if not date_str:
        return False
    try:
        sent_dt = parsedate_to_datetime(date_str)
        now = datetime.now(tz=timezone.utc)
        age = now - sent_dt.astimezone(timezone.utc)
        return age > timedelta(hours=STALE_HOURS)
    except Exception:
        return False


def poll(clear=False):
    cfg = load_env()
    if cfg is None:
        sys.exit(1)  # No credentials — silently skip

    host   = cfg.get("IMAP_HOST", "imap.gmail.com")
    user   = cfg.get("IMAP_USER", "")
    passwd = cfg.get("IMAP_PASS", "").replace(" ", "")  # strip spaces from app password

    if not user or not passwd:
        print("poll_steering_email: IMAP_USER or IMAP_PASS missing in .env", file=sys.stderr)
        sys.exit(2)

    try:
        conn = imaplib.IMAP4_SSL(host, timeout=20)
        conn.login(user, passwd)
        conn.select("INBOX")

        # Search for UNSEEN emails with "Re:" in subject (replies to agent emails).
        # We search broadly — any unseen reply is candidate steering context.
        _, data = conn.search(None, '(UNSEEN SUBJECT "Re:")')
        msg_ids = data[0].split()

        # Also check for unseen emails with "singlify" in subject (in case client
        # doesn't add Re: prefix, e.g. some mobile clients)
        _, data2 = conn.search(None, '(UNSEEN SUBJECT "singlify")')
        extra_ids = [m for m in data2[0].split() if m not in msg_ids]
        all_ids = msg_ids + extra_ids

        if not all_ids:
            conn.logout()
            sys.exit(1)

        # Find the most recent matching message
        best_id = None
        best_date = None
        for uid in all_ids:
            _, hdr_data = conn.fetch(uid, "(BODY[HEADER.FIELDS (DATE SUBJECT FROM)])")
            hdr_bytes = hdr_data[0][1]
            hdr_msg = email.message_from_bytes(hdr_bytes)
            date_str = hdr_msg.get("Date", "")
            try:
                dt = parsedate_to_datetime(date_str).astimezone(timezone.utc)
            except Exception:
                dt = datetime.min.replace(tzinfo=timezone.utc)
            if best_date is None or dt > best_date:
                best_date = dt
                best_id = uid

        # Fetch full message
        _, msg_data = conn.fetch(best_id, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        body = get_text_body(msg)
        steering = strip_reply_quoting(body)
        stale = is_stale(msg)

        if clear:
            conn.store(best_id, "+FLAGS", "\\Seen")

        conn.logout()

        if not steering:
            sys.exit(1)

        # Prepend staleness warning if needed
        if stale:
            prefix = (
                "[NOTE: This reply was sent more than 6 hours ago and may reference "
                "an earlier development cycle. Incorporate as steering context but "
                "note it may be partially out of date.]\n\n"
            )
            steering = prefix + steering

        print(steering)
        sys.exit(0)

    except Exception as e:
        print(f"poll_steering_email error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    poll(clear=clear)



def load_env():
    cfg = {}
    if not os.path.exists(ENV_FILE):
        return None
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def decode_str(s):
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    parts = decode_header(s)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(part)
    return "".join(out)


def get_text_body(msg):
    """Extract plain-text body from an email.Message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def strip_reply_quoting(body):
    """Remove typical email reply quoting (lines starting with >, On ... wrote:)."""
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", stripped):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def poll(clear=False):
    cfg = load_env()
    if cfg is None:
        # No credentials — silently skip
        sys.exit(1)

    host = cfg.get("IMAP_HOST", "imap.gmail.com")
    user = cfg.get("IMAP_USER", "")
    passwd = cfg.get("IMAP_PASS", "")
    if not user or not passwd:
        print("poll_steering_email: IMAP_USER or IMAP_PASS missing in .env", file=sys.stderr)
        sys.exit(2)

    try:
        conn = imaplib.IMAP4_SSL(host, timeout=15)
        conn.login(user, passwd)
        conn.select("INBOX")

        # Search for unseen messages with the tag in subject
        _, data = conn.search(None, f'(UNSEEN SUBJECT "{SUBJECT_TAG}")')
        msg_ids = data[0].split()

        if not msg_ids:
            conn.logout()
            sys.exit(1)

        # Read the most recent matching message
        latest_id = msg_ids[-1]
        _, msg_data = conn.fetch(latest_id, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        subject = decode_str(msg.get("Subject", ""))
        body = get_text_body(msg)
        steering = strip_reply_quoting(body)

        if clear:
            # Mark seen so we don't re-process
            conn.store(latest_id, "+FLAGS", "\\Seen")

        conn.logout()

        if steering:
            print(steering)
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"poll_steering_email error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    poll(clear=clear)
