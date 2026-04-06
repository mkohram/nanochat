# PROBE_V3_LOG.md

## Purpose
Fresh experiment log for the reduced MQAR/GDH lab harness after the recent probe simplification pass.

- Primary script:
  - `experiments/mqar_gdh_mps_lab.py`
- Primary launchers:
  - `experiments/run_probe_easy_mps.sh`
  - `experiments/run_probe_hard_mps.sh`
- Archived older GPU-oriented lab materials:
  - `experiments/archive/gpu-lab/`
- Date created: 2026-03-29
- Goal: keep a clean log for the minimal blindfolded MQAR probe line, separate from the older scan-probe history.

---

## Change policy for this track
- Keep the lab harness narrow and probe-local.
- Prefer explicit runtime ablations over unused architectural scaffolding.
- Remove dead probe code once it is clearly not part of the active experiment line.
- Log architectural ablations, launcher/default changes, and notable outcomes in this file.

---

## Experiment entries

### 2026-04-05 — Entry 14 (add direct read/write load instrumentation and GDH-off usefulness eval)
- Date/time: 2026-04-05
- Hypothesis:
  - Under the active normalized mixer, final sidecar norm-share is not a faithful proxy for routing load.
  - The probe needs direct write/load concentration metrics and an explicit causal usefulness check before further architecture changes.
- Change:
  1. Added direct routing instrumentation from actual attention weights:
     - write attention concentration from per-head `alpha`
     - write load stats from head-invariant routed mass `mean_h alpha`
     - read attention/load stats from read softmax weights
  2. Kept existing sidecar geometry stats, but clarified them as state geometry rather than routing-load metrics.
  3. Added eval-time GDH-off comparison pass:
     - same eval batch
     - same trunk
     - GDH read/write disabled
     - logs CE / top1 / MRR deltas versus normal eval
  4. Updated dashboard to show:
     - state-geometry charts under explicit state labels
     - direct write/read routing-load charts
     - GDH-on vs GDH-off usefulness metrics
- Notes:
  - This does not unroll training.
  - Extra cost is one additional eval forward at logging points for `arch=gdh`.
  - Training-step behavior is unchanged.

### 2026-04-05 — Entry 13 (add dedicated hard ClimbMix MPS launcher)
- Date/time: 2026-04-05
- Hypothesis:
  - Hard ClimbMix has become a recurring stability / realism check and deserves a dedicated launcher rather than ad hoc override strings.
- Change:
  1. Added:
     - `experiments/run_probe_hard_climbmix_mps.sh`
  2. The wrapper delegates to the active hard MPS launcher and applies ClimbMix-specific overrides:
     - `--data-source climbmix`
     - `--gdh-slots 32`
     - `--gdh-write-heads 16`
     - `--swa-window 0`
     - `--future-summary-horizon 0`
     - `--future-summary-lambda 0.0`
     - `--eval-batch-size 2`
     - `--log-every 5`
- Notes:
  - This intentionally inherits the current active hard-launcher defaults for the rest of the probe surface.
  - Extra flags passed to the wrapper still override these defaults.

### 2026-04-05 — Entry 12 (make normalized write mass head-invariant)
- Date/time: 2026-04-05
- Hypothesis:
  - The normalized state mixer should average write usage per token, not per write head.
  - Head count should change write factorization, not the total routing-mass budget contributed by a token.
- Change:
  1. Changed routed mass used by normalized state mixing from:
     - `sum_h alpha[t,h,r]`
     - to
     - `mean_h alpha[t,h,r]`
  2. Applied the same head-invariant routed mass convention to the write-cooloff usage trace.
- Notes:
  - With `gdh_write_heads=1`, this is a no-op.
  - With `gdh_write_heads>1`, each token now contributes total routed mass `1` rather than `H`.

### 2026-04-05 — Entry 11 (promote seeded write routing to active launcher default)
- Date/time: 2026-04-05
- Hypothesis:
  - The one-state bootstrap direction is now the active write-routing experiment line.
  - Canonical easy/hard launchers should default to `seeded` rather than permanent static routing.
- Change:
  1. Updated active MPS launchers to pass:
     - `--write-routing seeded`
  2. Files changed:
     - `experiments/run_probe_easy_mps.sh`
     - `experiments/run_probe_hard_mps.sh`
- Notes:
  - This changes launcher defaults only.
  - Lab argparse still supports:
     - `static`
     - `content`
     - `seeded`
     - `hybrid`
  - `seeded` means:
     - layer 0 routes from learned slot seed state
     - later layers route from previous-layer sidecar state only

### 2026-04-05 — Entry 10 (seeded write-routing mode for one-state bootstrap)
- Date/time: 2026-04-05
- Hypothesis:
  - Pure content-based write routing collapses because the slot state starts identically at zero.
  - A cleaner simplification than permanent static slot routing is to route from an evolving slot state that is bootstrapped from learned slot seeds only at layer 0.
- Change:
  1. Added new write-routing mode:
     - `--write-routing seeded`
  2. `seeded` now means:
     - layer 0 routing state = learned slot seed state `E_slots`
     - later-layer routing state = previous-layer sidecar state only
  3. Interpretation:
     - `E_slots` act as learned slot seed state for bootstrap, not as a permanent additive routing bias.
  4. Read path is intentionally unchanged for now so this isolates write-side behavior.
- Notes:
  - This keeps token-parallel training intact.
  - At layer 0, all tokens see the same learned slot seed table, but slots remain asymmetric because seed vectors differ by slot.
  - This is a write-side step toward a one-state `S` formulation without yet changing read semantics.

### 2026-03-31 — Entry 09 (minimal grad-accum knob for larger effective batches)
- Date/time: 2026-03-31
- Hypothesis:
  - On this MPS machine, larger effective batches should come from simple gradient accumulation rather than pushing microbatch size to the OOM cliff.
  - The probe can support this with minimal intrusion by accumulating multiple standard micro-steps before each optimizer step.
- Change:
  1. Added new flag:
     - `--grad-accum-steps`
  2. Training loop now:
     - runs `grad_accum_steps` micro-batches per optimizer step
     - divides each micro-step loss by `grad_accum_steps` before backward
     - calls `opt.step()` once after the accumulation loop
  3. Train-side logged scalars are now the mean across micro-steps for that optimizer step.
  4. Run labels now include:
     - `__ga=...`
  5. Startup logging now prints:
     - microbatch size
     - grad accumulation steps
     - effective batch size
     - effective tokens per optimizer step
- Notes:
  - This keeps eval behavior unchanged.
  - It is intentionally simpler than mainline distributed batch handling: no DDP-aware token math, just local micro-step accumulation.

### 2026-03-30 — Entry 08 (future-summary auxiliary head scaffold)
- Date/time: 2026-03-30
- Hypothesis:
  - The final sidecar state may benefit from a direct pressure to encode short-horizon future context rather than only supporting the answer-token CE path.
  - A small MLP trained to predict a summary of the next `N` tokens could encourage the state to think ahead.
- Change:
  1. Added a probe-local future-summary auxiliary head on the final sidecar state.
  2. The head uses slot-mean pooled final sidecar state at each token position.
  3. The target is the mean token embedding of the next `N` tokens.
  4. The auxiliary loss is cosine distance between predicted and target future summaries.
  5. Added new flags:
     - `--future-summary-horizon`
     - `--future-summary-lambda`
     - `--future-summary-hidden-mult`
  6. Run labels now include:
     - `__fs=<lambda>x<horizon>`
  7. Logged eval/train auxiliary scalars:
     - `eval_future_summary_loss`
     - `train_future_summary_loss`
- Notes:
  - This is currently implemented only for `arch=gdh`.
  - Setting either horizon or lambda alone is treated as invalid; both must be enabled together.
  - The target summary is detached from the current embedding table during loss computation.
- Validation:
  - `python3 -m py_compile experiments/mqar_gdh_mps_lab.py`
  - 2-step MQAR smoke test with `--future-summary-horizon 4 --future-summary-lambda 0.1` passed.

### 2026-03-30 — Entry 07 (remove capacity-stress enforcement from hard launchers)
- Date/time: 2026-03-30
- Hypothesis:
  - Capacity-stress enforcement is not a launcher responsibility.
  - Hard launchers should remain usable under slot-count overrides without forcing direct harness invocation.
- Change:
  1. Removed hard-coded `--enforce-capacity-stress` from:
     - `experiments/run_probe_hard_mps.sh`
     - `experiments/archive/gpu-lab/run_probe_hard.sh`
- Notes:
  - Capacity stress can still be requested explicitly via a direct override when desired.
  - This restores the intended launcher workflow for hard-probe ablations such as `--gdh-slots 16`.

### 2026-03-30 — Entry 06 (promote normalized state mixer to active launcher default)
- Date/time: 2026-03-30
- Hypothesis:
  - Raw accumulated write state still looks more collapse-prone than the routing-mass normalized variant in the current active probe line.
  - The normalized mixer is strong enough to become the default active MPS launcher setting while sum remains available as an explicit override.
- Change:
  1. Updated canonical active MPS launchers to pass:
     - `--state-mixer normalized`
  2. Files changed:
     - `experiments/run_probe_easy_mps.sh`
     - `experiments/run_probe_hard_mps.sh`
  3. This only changes launcher defaults.
     - The lab argparse still accepts both:
       - `--state-mixer sum`
       - `--state-mixer normalized`
     - Passing `--state-mixer sum` after the launcher still overrides the default.
- Notes:
  - Dense routing remains the active launcher default unless explicitly overridden.
  - This keeps the active probe surface aligned with current experiments without removing the sum ablation.

### 2026-03-30 — Entry 05 (normalized state-mixer ablation scaffold)
- Date/time: 2026-03-30
- Hypothesis:
  - Raw cumulative-sum sidecar accumulation may be amplifying collapse simply because heavily written slots accumulate unbounded magnitude.
  - A routing-mass normalized running-average mixer should test whether accumulation itself is the problem while preserving the token-parallel scan structure.
- Change:
  1. Added new probe flag:
     - `--state-mixer {sum,normalized}`
  2. `sum` keeps the current behavior:
     - sidecar update uses accumulated write deltas directly
  3. `normalized` now computes, per layer:
     - numerator = tokenwise scan of write deltas
     - denominator = tokenwise scan of per-slot routing mass
     - layer contribution = numerator / clamp(denominator, eps)
  4. Run labels now include:
     - `__mix=...`
  5. Dashboard compact config now shows:
     - `state_mixer`
- Notes:
  - This is intentionally a strong diagnostic ablation.
  - It changes the semantics of sidecar state from raw accumulated write content toward routing-mass normalized running averages.
  - The current implementation keeps the existing layerwise additive structure and the same scan primitive family.
- Next:
  - Run easy MQAR with `state_mixer=normalized` first.
  - Compare slot cosine collapse, participation ratio, and MRR against the dense default `sum` mixer.

### 2026-03-30 — Entry 04 (promote dense routing to active default)
- Date/time: 2026-03-30
- Hypothesis:
  - Hard top-k masking may be creating unnecessary dead-slot behavior, especially in early layers.
  - Dense static routing is a better active default until stronger evidence favors sparse routing again.
- Change:
  1. Updated canonical active MPS launchers to use dense routing by default:
     - `experiments/run_probe_easy_mps.sh`
     - `experiments/run_probe_hard_mps.sh`
  2. Default launcher arg changed from:
     - `--route-topk 4`
     - to
     - `--route-topk 0`
  3. Recent-write cooloff remains an explicit ablation knob on top of the new dense default.
- Notes:
  - This only changes the launcher defaults, not the lab argparse default.
  - Dense routing should be treated as the current active probe default pending comparison follow-through on easy and hard runs.
- Next:
  - Run easy dense-routing + cooloff as the next active reference run.
  - Compare layer-0 dead-slot behavior and collapse telemetry against the older sparse default.

### 2026-03-29 — Entry 03 (recent-write cooloff ablation scaffold)
- Date/time: 2026-03-29
- Hypothesis:
  - Current collapse may come less from narrow routing and more from repeatedly writing into recently hot slots.
  - A local, mechanism-level cooloff based on recent routing mass may reduce runaway slot reuse without adding an explicit geometry loss.
- Change:
  1. Added write cooloff knobs:
     - `--write-cooloff-lambda`
     - `--write-cooloff-rho`
  2. `_write_delta(...)` now returns both:
     - `delta`
     - final routing weights `alpha`
  3. In `_write_delta(...)`, when `write_cooloff_lambda > 0`:
     - compute a first-pass routing distribution
     - compute recent slot write-mass as head-summed routing mass per token/slot
     - run a leaky tokenwise scan over that recent usage with decay `write_cooloff_rho`
     - shift by one token to get prior recent usage only
     - subtract recent usage from routing logits before final slot selection
  4. Run labels now include:
     - `__wc=<lambda>@<rho>`
  5. Dashboard compact config now shows:
     - `write_cooloff_lambda`
     - `write_cooloff_rho`
- Notes:
  - This intervention is probe-local and keeps the existing parallel tokenwise structure.
  - It is intended as a more organic alternative to slot-geometry regularization losses.
  - It targets recent slot overuse, not total accumulated slot norm.
- Next:
  - Compare easy MQAR with `topk=4` and recent-write cooloff on/off.
  - Watch slot cosine max/p90, participation ratio, and MRR together.

### 2026-03-29 — Entry 02 (ClimbMix hook for probe-side collapse audits)
- Date/time: 2026-03-29
- Hypothesis:
  - Some apparent slot-collapse behavior may be specific to blindfolded MQAR geometry rather than a fully generic failure mode.
  - The reduced GDH probe should be runnable on a small slice of the same BOS-packed pretraining data used by the mainline base model, so collapse telemetry can be inspected on real text as well as MQAR.
- Change:
  1. Added probe data-source switch:
     - `--data-source {mqar,climbmix}`
  2. Hooked `data_source=climbmix` into the existing mainline data path.
     - Reused `nanochat.tokenizer.get_tokenizer()`
     - Reused `nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit(...)`
     - Train batches now come from ClimbMix `train` split when requested.
     - Eval batch now comes from ClimbMix `val` split when requested.
  3. Added ClimbMix loader runtime knobs for the probe harness:
     - `--base-tokenizer-threads`
     - `--base-tokenizer-batch-size`
     - `--base-buffer-size`
  4. For `data_source=climbmix`, the probe now overrides `vocab_size` to the tokenizer vocabulary size so token ids from the real dataset fit the trunk embedding table.
  5. Run labels now include `__data=...` so MQAR vs ClimbMix runs are easy to distinguish in logs/dashboard.
  6. Dashboard compact config now shows `data_source`.
- Command examples:
  - MQAR easy default:
    - `bash experiments/run_probe_easy_mps.sh`
  - ClimbMix easy-shaped probe:
    - `bash experiments/run_probe_easy_mps.sh --data-source climbmix --swa-window 0`
  - ClimbMix hard-shaped blindfolded probe:
    - `bash experiments/run_probe_hard_mps.sh --data-source climbmix`
- Notes:
  - MQAR-specific geometry checks remain enforced only for `data_source=mqar`.
  - The probe still uses the reduced GDH path and existing collapse telemetry on top of the mainline trunk.
  - This change is intended as instrumentation / comparison support before trying stronger anti-collapse interventions.
- Verdict:
  - The probe can now compare synthetic recall geometry against real pretraining text without adding a second custom dataset path.
- Next:
  - Run short ClimbMix GDH and baseline checks.
  - Compare slot participation ratio / max-share trajectories between MQAR and ClimbMix.

### 2026-03-29 — Entry 01 (probe minimization + write-routing ablation scaffold)
- Date/time: 2026-03-29
- Hypothesis:
  - The probe should be simplified down to the active reduced GDH path only.
  - The slot-usage balancing loss is unnecessary for the current lab line.
  - The main write-routing ablation should isolate three routing sources cleanly:
    - static learned slot addresses,
    - pure content-based routing from current sidecar state,
    - hybrid static + content routing.
  - Static routing should remain the stable default.

- Change:
  1. Removed unused probe loss machinery.
     - Deleted usage-balance loss from both lab harnesses.
     - Removed `--usage-balance-lambda` from probe argparse.
     - Removed `__ubal=...` from run labels.
     - Removed zero-only `eval_usage_loss` / `train_usage_loss` plumbing from probe records and dashboard types.
  2. Removed dead probe write-path outputs.
     - Simplified `_write_delta_and_alpha(...)` into `_write_delta(...)`.
     - Deleted unused returned routing tensors after usage-balance removal.
  3. Cleaned stale launcher drift and naming.
     - Fixed stale older-GPU launcher drift before archiving that lab path.
     - Launcher log filenames now reflect the effective `--steps` value instead of hardcoded `5k`.
  4. Added explicit write-routing ablation flag.
     - New flag in both harnesses:
       - `--write-routing {static,content,hybrid}`
     - Modes:
       - `static`: current learned-slot-address routing (`E_slots` only)
       - `content`: route using current `sidecar_prev` only
       - `hybrid`: sum static and content logits before softmax
     - Run labels now include `__wroute=...`.
  5. Added dashboard/config visibility for the new routing ablation.
     - Dashboard compact config now shows `write_routing`.
  6. Added extra collapse telemetry.
     - New layerwise metric: `slot_cos_p90` (p90 off-diagonal slot cosine).
     - Dashboard collapse section now shows one combined graph for:
       - max off-diagonal slot cosine
       - p90 off-diagonal slot cosine
  7. Made `static` explicit in the canonical active MPS probe launchers.
     - `run_probe_easy_mps.sh`
     - `run_probe_hard_mps.sh`
     - active launchers now pass `--write-routing static`
     - older-GPU launchers were later archived to `experiments/archive/gpu-lab/`

- Commands / runs used during this pass:
  - Easy blind GDH, static routing:
    - `bash experiments/run_probe_easy_mps.sh --write-routing static`
  - Easy blind GDH, content routing:
    - `bash experiments/run_probe_easy_mps.sh --write-routing content`
  - Easy blind GDH, hybrid routing:
    - `bash experiments/run_probe_easy_mps.sh --write-routing hybrid`
  - Easy blind baseline:
    - `bash experiments/run_probe_easy_mps.sh --arch baseline`
  - Easy non-blind baseline:
    - `bash experiments/run_probe_easy_mps.sh --arch baseline --swa-window 0`
  - Hard non-blind baseline:
    - `bash experiments/run_probe_hard_mps.sh --arch baseline --swa-window 0`
  - Default hard blind GDH:
    - `bash experiments/run_probe_hard_mps.sh`

- Key observations:
  - Removing usage-balance loss looked favorable enough to drop it from the active probe line.
  - Pure content write routing showed immediate symmetry/collapse behavior consistent with zero-initialized sidecar state.
    - At sequence start, content-only routing has no slot differentiation.
  - Hybrid routing behaved much closer to static than to content-only in early probe behavior.
  - Static routing remained the sensible default after the ablation scaffold was added.
  - Even the hard non-blind baseline did not clearly solve the task under the current small-model/5k-step setup, which reinforces that the hard blindfolded probe is a strong stress test.

- Notable implementation issue encountered:
  - Initial `slot_cos_p90` implementation crashed on MPS because `torch.quantile(...)` rejected the input dtype/device combination during eval.
  - Fixed by computing the p90 quantile from a CPU `float32` copy of the off-diagonal slot-cosine tensor.

- Validation:
  - `experiments/archive/gpu-lab/mqar_gdh_lab.py --help` passed before older-GPU lab archival.
  - `experiments/mqar_gdh_mps_lab.py --help` passed after cleanup.
  - `cd experiments/probe-dashboard && npm run build` passed after dashboard updates.
  - Easy blind static probe relaunched successfully after the p90 fix.

- Verdict:
  - The probe line is now materially cleaner and closer to the actual active experiment surface.
  - Static write routing remains the default and current source of truth.
  - Content-only routing is a useful negative-control ablation.
  - Hybrid routing is implemented and ready for controlled comparison runs.

- Current default state of the minimal probe line:
  - no usage-balance loss
  - no dead routing-loss plumbing
  - explicit write-routing ablation support
  - canonical launchers pinned to `--write-routing static`
  - extra collapse telemetry via max + p90 off-diagonal slot cosine

- Next:
  - Run controlled static vs hybrid comparisons on easy and hard blindfolded probes.
  - Decide whether hybrid deserves deeper tuning or whether static should remain the sole active write path.

---

### Entry template
- Date/time:
- Hypothesis:
- Change:
- Command:
- Key outputs:
- Verdict:
- Next:
