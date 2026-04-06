# EXPERIMENTATION_GUIDE

Internal runbook for continuing GDH probe experiments consistently after context compaction.

## Purpose

This file records how experiments are currently being launched, stopped, and monitored.

Use this together with:
- `experiments/ACTIVE_GDH_PROBE_ARCHITECTURE.md` for the current established architecture
- `experiments/PROBE_V3_LOG.md` for architectural/default changes and experiment history

## Current launcher surface

Use launcher scripts first. Do not invoke `experiments/mqar_gdh_mps_lab.py` directly unless a launcher truly does not exist.

Primary active launchers:
- `experiments/run_probe_easy_mps.sh`
- `experiments/run_probe_hard_mps.sh`
- `experiments/run_probe_hard_climbmix_mps.sh`
- `experiments/run_probe_dashboard.sh`

Current active defaults inherited from the launchers:
- `write_routing=seeded`
- `state_mixer=normalized`
- `route_topk=0`
- `read_mute_gate=off`
- `gdh_use_write_brain=off`
- `future-summary=off` in the established architecture

## Hard rules

1. Prefer launchers with overrides.
   - Good:
     - `bash experiments/run_probe_hard_mps.sh --gdh-write-heads 8`
   - Avoid:
     - direct long `python experiments/mqar_gdh_mps_lab.py ...`

2. Kill the current probe before starting a new one unless parallelism is explicitly desired.

3. Keep one active probe at a time by default.

4. Always check the active log pointer after launch:
   - `experiments/out/probe_current_log_path.txt`

5. Be aware that the run label is still incomplete.
   - In particular, it may not encode `gdh_write_heads`.
   - When head count matters, preserve the exact CLI used to launch the run.

## Canonical stop sequence

Use this before launching a new probe:

```bash
pkill -f 'experiments/mqar_gdh_mps_lab.py' 2>/dev/null || true
sleep 2
ps -axo pid=,command= | rg 'experiments/(run_probe_(easy|hard|hard_climbmix)_mps\.sh|mqar_gdh_mps_lab\.py)' -n -S || true
```

Notes:
- `pkill` targets the Python harness.
- wrapper bash processes may linger briefly; that is normal.

## Canonical background launch pattern

Use this pattern for runs started from the agent:

```bash
nohup bash experiments/run_probe_...sh [overrides...] > /tmp/<short_name>.out 2>&1 &
```

Then check:

```bash
sleep 12
cat experiments/out/probe_current_log_path.txt
tail -n 80 /tmp/<short_name>.out
```

## Canonical launcher commands

### Easy MQAR
```bash
bash experiments/run_probe_easy_mps.sh
```

Common overrides:
```bash
bash experiments/run_probe_easy_mps.sh --gdh-write-heads 8
bash experiments/run_probe_easy_mps.sh --write-routing static
bash experiments/run_probe_easy_mps.sh --log-every 1
bash experiments/run_probe_easy_mps.sh --steps 1000
```

### Hard MQAR
```bash
bash experiments/run_probe_hard_mps.sh
```

Common overrides:
```bash
bash experiments/run_probe_hard_mps.sh --gdh-write-heads 8
bash experiments/run_probe_hard_mps.sh --gdh-slots 16
bash experiments/run_probe_hard_mps.sh --write-routing static
bash experiments/run_probe_hard_mps.sh --log-every 5
```

### Hard ClimbMix
Use the dedicated wrapper:
```bash
bash experiments/run_probe_hard_climbmix_mps.sh
```

Current wrapper defaults:
- `data_source=climbmix`
- `gdh_slots=32`
- `gdh_write_heads=16`
- `swa_window=0`
- `future_summary_horizon=0`
- `future_summary_lambda=0.0`
- `eval_batch_size=2`
- `log_every=5`

Common overrides:
```bash
bash experiments/run_probe_hard_climbmix_mps.sh --steps 1000
bash experiments/run_probe_hard_climbmix_mps.sh --gdh-write-heads 8
bash experiments/run_probe_hard_climbmix_mps.sh --batch-size 32
bash experiments/run_probe_hard_climbmix_mps.sh --grad-accum-steps 2
```

## Monitoring

### Active log path
```bash
cat experiments/out/probe_current_log_path.txt
```

### Tail the active log
```bash
tail -n 80 "$(cat experiments/out/probe_current_log_path.txt)"
```

### Check whether a probe is alive
```bash
ps -axo pid=,etime=,command= | rg 'experiments/(run_probe_(easy|hard|hard_climbmix)_mps\.sh|mqar_gdh_mps_lab\.py)' -n -S || true
```

### Inspect live JSON quickly
```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/out/probe_live.json')
obj = json.loads(p.read_text())
last = obj.get('last', {})
for k in [
    'step',
    'eval_ce',
    'eval_acc_top1',
    'eval_mrr',
    'state_slot_cos_l_last',
    'eval_mem_delta_ce',
    'write_load_max_share_layers',
    'read_load_max_share_layers',
]:
    print(f'{k}: {last.get(k)}')
PY
```

### Current high-priority metrics
Treat these as distinct things:

1. State geometry
- `state_slot_cos_*`
- `state_effective_slots_layers`
- `state_max_share_layers`
- `state_participation_ratio_layers`

2. Actual write/read routing behavior
- `write_load_*`
- `write_attn_*`
- `read_load_*`
- `read_attn_*`

3. Causal memory usefulness
- `eval_gdh_off_ce`
- `eval_gdh_off_acc_top1`
- `eval_mem_delta_ce`
- `eval_mem_delta_acc_top1`

Important:
- under `state_mixer=normalized`, state norm-share is not a faithful proxy for routing load
- healthy-looking state usage can coexist with collapsed routing load

## Dashboard

Start dashboard:
```bash
bash experiments/run_probe_dashboard.sh
```

Default URL:
- `http://127.0.0.1:4173`

Notes:
- dashboard reads `experiments/out/probe_live.json`
- if the UI looks empty, first verify `probe_live.json` is valid and not just very large

## Current established architecture boundary

When describing the architecture or handing it to an auditor, do not include inactive/optional features as if they are active defaults.

Current established architecture excludes:
- future-summary auxiliary loss
- write cooloff
- write brain
- sparse top-k routing
- read-mute gate

If those are turned on for a run, treat them as explicit ablations.

## Current simplification direction

The active write-routing direction is:
- `seeded`

Meaning:
- layer 0 write routing uses learned slot seed state
- later layers route from previous-layer sidecar state only

This is the current simplification path toward a one-state memory story.

## Practical notes

1. `--log-every 1` is often too slow on heavier runs.
   - Prefer `5` or `50` unless dense visibility is needed.

2. For ClimbMix stability checks, use the wrapper rather than rebuilding the override string from memory.

3. For head-count experiments, remember that run labels may not include the head count.
   - Mention the exact CLI in summaries.

4. Use `--steps` through the launcher.
   - launchers name logs using the effective `steps` value.

5. If a user asks for the current active architecture, refer to:
   - `experiments/ACTIVE_GDH_PROBE_ARCHITECTURE.md`

## Minimal restart checklist

When switching experiments:
1. kill current probe
2. launch a launcher script with overrides
3. inspect `/tmp/...out`
4. inspect `probe_current_log_path.txt`
5. tail the active log
6. confirm live dashboard is following the new run
