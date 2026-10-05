# Long branch replacement: preregistered pilot

This is a new, bounded experiment authorized after the earlier diagnostic was paused. It does not restart that queue. Original core and vendor files remain unchanged.

## task_plan

1. Implement three model arms and a test-label-safe frozen-data adapter; verify initialization, signs, padding, gradients and smoke training.
2. Commit and publish code/configuration before full training.
3. Run two datasets, seed 42, three arms (six runs); audit metrics and replay best checkpoints on the training GPU.
4. Publish results and stop if either dataset fails the gate. Only if both pass, allow seeds 43/44 (12 additional runs), then report matched results.

## Frozen comparison

- full: original ShortGINE + LongPoly, original SAG pooling, antisymmetric delta head.
- short: identity replaces LongPoly, retaining ShortGINE, sigmoid gate and residual. No trainable long module remains.
- cross: shared bidirectional CrossInteraction replaces LongPoly at EACH filter; local features of both molecules are computed before either update. Attention has its existing residual and LayerNorm. No attention after the full encoder.
- Reuse existing trainer, optimizer, schedule, MSE, feature generation and structural top-1 reference policy. Retained weights, pooling and head share the original initialization. Cross adds capacity; results cannot isolate attention from capacity/normalization.
- CHEMBL234_Ki and CHEMBL244_Ki, original frozen split/pairs, seed 42. Hidden 256, three filters, four heads, batch 32, maximum 100 epochs, patience 15, warmup 10, AdamW and cosine settings copied unchanged from interaction_seed42.json. Best checkpoint by validation Overall MSE only.
- CSV byte hashing checks identity; only smiles/split are parsed outside the development rows. Test activity and test cliff labels are never parsed; no test inference.

## Gate fixed before results

Both datasets must have cross Cliff RMSE <= 0.95 times BOTH full and short controls, and cross Overall RMSE <= 1.01 times the better control. Additionally cross Cliff RMSE must be <= 1.05 times the archived direct seed-42 baseline. These pragmatic exploration thresholds are not paper parameters. Any failure means No-Go: no extra seeds, tasks, FPPool or loss changes. Smoke runs do not count as performance evidence. Maximum full budget is 18 runs; no automatic score-driven retries.

## notes

Earlier post-encoder attention did not support expansion. This pilot tests a different implementation location; the earlier diagnostics establish no causal mechanism. Report parameter counts, actual epochs/time and standardized activity units. A seed-42 gate failure is a bounded negative result, not a proof that all cross attention is impossible.

## Completed screen

All six full-budget seed-42 runs and GPU checkpoint replays completed. No-Go: neither task passes the frozen gate; no expansion. See [Chinese results](结果报告.md), [aggregate results](results.md), and screen_manifest/audit/decision JSON records.
