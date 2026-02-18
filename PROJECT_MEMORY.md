# Project Memory: nanochat (GTX 1060 3GB safe settings)

Last updated: 2026-02-18

## Known-safe training config (this machine)

Use this for stable `local_base_train` runs on NVIDIA GTX 1060 3GB:

- `--device-batch-size=1`
- `--total-batch-size=256`
- `--num-iterations=10000` (or as needed)
- `--core-metric-every=-1` (disable CORE during training)
- `--eval-every=-1` (disable val eval during training; prevents end-of-run eval OOM risk)
- `--sample-every=-1` (optional, reduces extra memory/time overhead)

### Why

Larger micro-batch settings (e.g. `--device-batch-size=32`) OOM on this GPU.
Typical idle VRAM is ~250 MB used, so OOMs here are config-related, not background usage.

## Canonical safe launch command

```bash
source .venv/bin/activate
python -m scripts.local_base_train \
  --run baseline-d10-s256-10k-safe-<timestamp> \
  --num-iterations=10000 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --core-metric-every=-1 \
  --eval-every=-1 \
  --sample-every=-1
```

## Note

- `scripts/base_train.py` and `scripts/base_eval.py` are kept clean for scale/upstream behavior.
- `scripts/local_base_train.py` and `scripts/local_base_eval.py` are the local GTX 1060 paths.
- If you re-enable evals, ensure checkpointing behavior matches intent (current script saves at end, after eval blocks).
