# SPDX-License-Identifier: GPL-2.0-or-later
# scripts/load_secrets.sh — source this file before any Phase G publish step.
#
# Usage:
#   source ~/Singlet-AI/singlet-gpu/scripts/load_secrets.sh
#
# Loads Supabase credentials from ~/.config/singlet/supabase.env (chmod 600,
# outside the repo). The repo only references env var names, never values.
#
# If you want the orchestrator's SLURM scripts to have these too, add this
# line to your ~/.bashrc or ~/.bash_profile:
#   [ -f ~/.config/singlet/supabase.env ] && source ~/.config/singlet/supabase.env

set +u
SECRETS_FILE="${SINGLET_SECRETS_FILE:-$HOME/.config/singlet/supabase.env}"

if [ -r "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
else
    echo "[load_secrets] missing or unreadable: $SECRETS_FILE" >&2
    echo "[load_secrets] Create it with:" >&2
    echo "  mkdir -p ~/.config/singlet && chmod 700 ~/.config/singlet" >&2
    echo "  cat > ~/.config/singlet/supabase.env <<EOF" >&2
    echo "export SUPABASE_URL=\"...\"" >&2
    echo "export SUPABASE_SERVICE_KEY=\"...\"" >&2
    echo "export SUPABASE_ANON_KEY=\"...\"" >&2
    echo "EOF" >&2
    echo "  chmod 600 ~/.config/singlet/supabase.env" >&2
fi
set -u 2>/dev/null || true
