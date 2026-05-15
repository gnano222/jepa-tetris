# Design: Counterfactual + FiLM + Teacher-Forced Multi-Step

**Date:** 2026-05-14  
**Status:** Approved  
**Goal:** Combine counterfactual learning with FiLM conditioning and teacher-forced multi-step training to beat film-100k on both multistep prediction metrics and action causality metrics.

---

## Background

Two improvements have been developed independently:

**FiLM conditioning** (`jepa-exp-film-100k.pt`) makes the action signal unconditional and unavoidable at every transformer layer via per-layer `γ/β` modulation. Result: cos@16 went from 0.02 → 0.93, DROP MSE halved. Trained with teacher-forced H=4, single-action objective.

**Counterfactual learning** (`jepa_cf_compare_5x.pt`) forces the encoder to preserve all-action information from every starting state — an implicit contrastive regularizer. Result: at ≥3× compute, wins M1 (action retrieval) by 10pp over single-action. Trained with single-step CF fanout only, pre-FiLM architecture.

These have never been combined. They attack complementary weaknesses:
- FiLM fixes how the **predictor uses** the action signal
- CF fixes what the **encoder retains** to support all possible actions

---

## Training Objective

The new combined loss per step:

```
L = L_CF@t=0  +  L_TF@t=1..H  +  VICReg
```

**L_CF@t=0** — counterfactual fanout from real `z_0`:
```
(1/A) · Σ_a MSE( predictor(encoder(s_0), a),  stop_grad(target_encoder(s'_0^a)) )
```
Forces `z_0` to support predictions for all 4 actions simultaneously.

**L_TF@t=1..H** — teacher-forced multi-step on executed action (H=4):
```
(1/H) · Σ_{t=1}^{H} MSE( predictor(encoder(s_t), a_t),  stop_grad(target_encoder(s_{t+1})) )
```
Uses real encoder outputs at each step (not chained predictor outputs) — same signal responsible for film-100k's long-horizon stability.

**FiLM** applies unconditionally at every predictor call in both terms.

**Total predictor calls per step:** 4 (CF fanout) + 4 (TF chain) = 8, vs film-100k's 4 TF calls. The 50k-step run is therefore compute-parity with film-100k (50k × 8 = 400k predictor calls = 100k × 4). The 100k-step run is 2× parity.

---

## Code Changes

### New flag

`--cf-multistep` (boolean, default off). When set alongside `--counterfactual`, activates the combined CF@t=0 + TF@t=1..H objective. Existing `--counterfactual` behaviour (single-step fanout, optional `--ar-weight`) is fully preserved when `--cf-multistep` is not set.

### New code path in `train.py`

Inside the `if args.counterfactual:` block, a new branch triggered by `args.cf_multistep`:

```python
if args.cf_multistep:
    batch = buf.sample_rollout(args.batch_size, H, rng=rng)
    s0 = torch.from_numpy(batch["s0"]).to(device)                   # (B, *state)
    actions_executed = torch.from_numpy(batch["actions_executed"]).to(device)  # (B, H)
    next_states_k = torch.from_numpy(batch["next_states_k"]).to(device)       # (B, H, A, *state)

    # 1. CF fanout at t=0
    next_states_t0 = next_states_k[:, 0]                            # (B, A, *state)
    cf_out = counterfactual_step_loss(
        s0=s0, next_states=next_states_t0,
        encoder=encoder, target_encoder=target_encoder,
        action_encoder=action_encoder, predictor=predictor,
    )
    mse_cf = cf_out["mse"]
    z_pred_all = cf_out["z_pred_all"]

    # 2. Teacher-forced on executed chain t=1..H
    # on_policy: (B, H, *state) — the real next-state at each step
    idx = actions_executed.view(B, H, 1, *([1]*len(state_shape)))
    idx = idx.expand(B, H, 1, *state_shape)
    on_policy = next_states_k.gather(2, idx).squeeze(2)             # (B, H, *state)

    # TF inputs: [s0, s1, ..., s_{H-1}]
    # TF targets: [s1, s2, ..., s_H]
    tf_inputs = torch.cat([s0.unsqueeze(1), on_policy[:, :H-1]], dim=1)  # (B, H, *state)
    tf_inputs_flat = tf_inputs.reshape(B * H, *state_shape)
    z_tf_in = encoder(tf_inputs_flat)                               # (B*H, N, D)
    with torch.no_grad():
        z_tf_tgt = target_encoder(on_policy.reshape(B * H, *state_shape))  # (B*H, N, D)
    a_emb_tf = action_encoder(actions_executed.reshape(B * H))
    z_tf_pred = predictor(z_tf_in, a_emb_tf)
    mse_tf = F.mse_loss(z_tf_pred, z_tf_tgt)

    mse = mse_cf + mse_tf
    z_for_vic = encoder(s0)
    z_pred_log = z_pred_all[:, 0]
    with torch.no_grad():
        z_target_log = target_encoder(next_states_t0[:, 0])
```

**`gather_on_policy` note:** The 4-line gather pattern above is equivalent to what the existing AR path does for `target_states_chain`. No new helper needed.

**Checkpoint format:** unchanged. The `--cf-multistep` flag is saved in `args` and reconstructed by `load_jepa()`.

**No other files change.** Predictor, encoder, action encoder, eval scripts, causality diagnostic — all unchanged.

---

## Experiment Setup

### Buffer

`data/cf_train_100k.npz` — existing 100k-row CF buffer. No new data collection.

### Training runs

| run name | steps | key flags | compute vs film-100k | checkpoint |
|---|---|---|---|---|
| cf-film-50k | 50 000 | `--counterfactual --cf-multistep --predictor-film --encoder-two-scale --encoder-stride-stages 2 --horizon-h 4 --batch-size 256 --ar-weight 0` | parity (400k predictor calls = film-100k's 400k) | `checkpoints/jepa-cf-film-50k.pt` |
| cf-film-100k | 100 000 | same, `--steps 100000` | 2× parity (800k predictor calls) | `checkpoints/jepa-cf-film-100k.pt` |

Both: `--seed 0`, separate `--run` names for timestamped results.

### Eval-only run (no training)

```bash
python scripts/causality_diagnostic.py \
    --jepa checkpoints/jepa-exp-film-100k.pt \
    --out results/causality_film100k.json
```

Fills the missing FiLM-without-CF cell in the comparison matrix.

### Full comparison matrix

| model | M1 | M2 | M4 | cos@16 | DROP MSE@1 |
|---|---|---|---|---|---|
| CF only @ 5× (`jepa_cf_compare_5x.pt`) | 0.912 | 0.861 | 0.322 | 0.978 | — |
| FiLM only (`jepa-exp-film-100k.pt`) | **eval** | **eval** | **eval** | 0.9309 | 0.0678 |
| CF+FiLM @ 50k (parity) | **train** | **train** | **train** | **train** | **train** |
| CF+FiLM @ 100k (2× parity) | **train** | **train** | **train** | **train** | **train** |

---

## Evaluation Plan

Run on both new checkpoints after training:

1. **`scripts/multistep_accuracy.py`** — cos@k and MSE@k at k=1,2,4,8,16. Per-action MSE breakdown. Primary metric for beating film-100k.

2. **`scripts/causality_diagnostic.py`** — M1, M2, M4. Primary metric for beating the CF study.

### Success criteria

| metric | target | source |
|---|---|---|
| cos@16 | > 0.9309 | film-100k |
| DROP MSE@1 | < 0.0678 | film-100k |
| M1 action retrieval | > 0.927 | CF @ 3× |
| M2 distance calibration | > 0.879 | CF @ 3× |
| M4 no-op recognition | < 0.309 | single-action @ 5× |

### Key things to watch

- **M4 with FiLM**: FiLM has never been evaluated on causality metrics. If FiLM alone already scores well on M1/M2, that changes the interpretation of the CF+FiLM result.
- **DROP@k at long horizon**: No conditioning variant has closed the 50× gap between DROP and movement actions. CF's contrastive regularizer may help here specifically, since DROP produces the most distinct counterfactual outcome of any action.
- **The 50k parity run**: if CF+FiLM already wins at parity compute, the combination is genuinely superior at equal cost. If it only wins at 100k steps (2× parity), the story is "more compute helps" rather than "CF+FiLM is better."

---

## Hypothesis

FiLM gives the predictor the *capacity* to differentiate actions strongly at every layer. CF gives the encoder the *training signal* to actually preserve that action-differentiating information. Together they should be additive: FiLM improves cos@k (especially k≥8), CF improves M1/M2, and the combination wins both simultaneously — something neither achieves alone.
