# Pilot Runbook — 1,000-Sample Human 10x Droplet Pilot

**Freeze tag**: `v0.3.0-pilot-freeze`
**Estimated wall-clock**: 2–5 days at MAX_QT=200
**Cluster**: Clipper HPC (SLURM)
**Reference**: GRCh38-2024-A (see `state/reference-manifest-v1.yaml`)

## Pre-flight checklist

Before launching the pilot, all of these must be green:

- [ ] On `v0.3.0-pilot-freeze` tag (`git describe --exact-match HEAD` returns the tag)
- [ ] Clean working tree (`git status` is empty)
- [ ] Binary built from this tag: `build/src/pipeline/singlet --help` runs cleanly
- [ ] Binary sha256 recorded in `state/pilot-binary-sha256.txt`
- [ ] All cpp tests pass: `cd build && ctest --output-on-failure`
- [ ] All python tests pass: `pytest tests/python`
- [ ] Reference verification passes: `scripts/verify_reference.py --ref-base $SINGLET_REF_BASE` returns 0
- [ ] Reference manifest sha256 captured and recorded
- [ ] Pilot selection exists: `scripts/pilot_1000.tsv` (1,000 lines + 1 header)
- [ ] Pilot manifest exists: `scripts/pilot_1000_manifest.json`
- [ ] Disk budget OK: `df -h /mnt/projects/debruinz_project` shows ≥ 5 TB free
- [ ] Orchestrator state cleared from prior run: `rm -f orchestrator_state.json`
- [ ] Slurm queue empty for this user: `squeue -u $USER` returns no rows
- [ ] No competing high-priority jobs on shared partitions

## Launch

```bash
cd /mnt/home/debruinz/Singlet-AI
export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/refs
export SINGLET_BINARY=$PWD/singlet/build/src/pipeline/singlet
export SINGLET_GIT_SHA=$(cd singlet && git rev-parse HEAD)
export SINGLET_REF_MANIFEST_SHA=$(sha256sum singlet/state/reference-manifest-v1.yaml | awk '{print $1}')

# Sanity: every variable populated
env | grep ^SINGLET_

# Launch via clipper orchestrator
python3 scripts/clipper_orchestrate.py \
  --manifest scripts/pilot_1000.tsv \
  --phase pilot \
  --max-qt 200 \
  --output-root /mnt/projects/debruinz_project/singlet_pipeline/results/pilot-1k-$(date +%Y%m%d) \
  --git-sha $SINGLET_GIT_SHA \
  --ref-manifest-sha $SINGLET_REF_MANIFEST_SHA \
  2>&1 | tee logs/pilot-1k-launch.log
```

## Monitor

```bash
# Headline status (refresh every 60s)
watch -n 60 'python3 scripts/monitor_loop.sh | head -40'

# Live log tail
tail -F logs/pilot-1k-launch.log logs/clipper_orchestrate_*.log

# SLURM queue
watch -n 30 'squeue -u $USER -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R" | head -50'

# Disk usage
df -h /mnt/projects/debruinz_project
```

## Common failure modes + triage

| Symptom | Likely cause | Action |
|---|---|---|
| Job stuck in PENDING > 1h | Cluster queue contention | Wait; do not resubmit |
| `singlet: command not found` in job log | $PATH not exported into SLURM env | Fix orchestrator submission template; do not retag |
| `STAR exited 137` | OOM, hit memory tier ceiling | Bump tier in catalog row; retry (data-driven, not pipeline-driven) |
| `NCBI download timeout` | Transient SRA fetch failure | Auto-retry built into orchestrator; if 3 retries fail → mark `fail_download` |
| `summary.json` missing `git_sha` | Code didn't pick up env var | Bug: stop, root-cause, hotfix branch |
| Output `.1pz` decode roundtrip fail | Codec bug | Bug: stop, root-cause, hotfix branch |
| > 15% samples in `align_low_map` | Reference mismatch or wrong protocol detection | Stop, investigate before continuing |
| > 20% samples in `align_zero_cells` | Cell-calling regression | Stop, compare against regression baseline (C8 samples) |
| Reference verification fails mid-run | Reference file mutated on disk | Stop everything, restore, never re-run from compromised refs |

## Abort

If the pilot must be stopped:

```bash
# Cancel all queued jobs (preserves running ones — let them finish or be killed)
scancel -u $USER --state=PENDING

# Cancel everything immediately
scancel -u $USER

# Mark orchestrator paused (it will not auto-resubmit)
touch orchestrator_state.paused
```

Captured outputs from completed samples remain valid. The orchestrator's `orchestrator_state.json` records which samples completed; a resume is `python3 scripts/clipper_orchestrate.py --resume`.

## Post-run aggregation

After the last job exits:

```bash
# Aggregate all summary.json files into a single parquet
python3 scripts/aggregate_pilot.py \
  --results-root /mnt/projects/debruinz_project/singlet_pipeline/results/pilot-1k-* \
  --output pilot_summary.parquet

# Validate every summary.json against schema 1.1
python3 scripts/validate_summary_schema.py \
  --parquet pilot_summary.parquet \
  --schema state/summary-schema-v1.1.json

# Build the QC dashboard
python3 scripts/build_pilot_qc_dashboard.py \
  --parquet pilot_summary.parquet \
  --output pilot_qc_report.html

# Spot-check regression against C8 (previously-completed) subset
python3 scripts/regression_check.py \
  --pilot pilot_summary.parquet \
  --baseline /mnt/projects/debruinz_project/cellarium/pipeline/quant/ \
  --output pilot_regression.parquet
```

## Go / no-go decision

Open `pilot_qc_report.html` and walk through `state/pilot-success-criteria.md` line by line. All criteria must pass.

- **Pass**: tag `v0.3.0-production` on the same commit; launch full 56K campaign with no code changes
- **Fail**: classify failure, follow `state/pilot-success-criteria.md § "Failure response"`

## Full-campaign launch (only if pilot passes)

```bash
cd /mnt/home/debruinz/Singlet-AI
git -C singlet checkout v0.3.0-production    # same commit as v0.3.0-pilot-freeze
# (binary already built; reuse)

# Phase 1: waves of 10K samples each
python3 scripts/build_wave.py --phase 1 --size 10000 --exclude scripts/pilot_1000.tsv
python3 scripts/clipper_orchestrate.py --manifest scripts/wave_1_10k.tsv --phase wave1 --max-qt 200
# … wave 2, wave 3 …
```

No source-tree changes happen between pilot and waves. If a change is needed, that's a freeze break — full STOP and re-pilot.
