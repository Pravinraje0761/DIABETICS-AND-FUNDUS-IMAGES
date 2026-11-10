# Stochastic-pass predictions with the linear probe made deterministic

Review finding (2026-10-21): `scripts/phase8_mc_dropout.py` rebuilt every RETFound
checkpoint with the backbone builder's default drop-path rate (0.2). The linear
probe was *trained* with drop_path = 0 (frozen backbone), so its files in
`outputs/phase8/mc/` injected stochastic depth it never saw in training. That is
not the intended sensitivity analysis ("keep active only the layers that were
stochastic during training").

The intended T-pass output of a model with no training-time stochastic layer is
its single-pass output. This folder therefore holds:
- EfficientNet-B3, ResNet-50, RETFound full fine-tune: copied unchanged from
  `outputs/phase8/mc/` (T = 20, head dropout / stochastic depth active);
- RETFound linear probe: copied from `outputs/phase6/` (single pass).

`scripts/phase8_mc_dropout.py` is now fixed to respect each checkpoint's own
drop_path; re-running it reproduces this substitution.
Results: `outputs/phase23/*_mc_lpdet.*`.
