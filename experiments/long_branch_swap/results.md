# LongPoly replacement: seed-42 screen

Two development datasets; no test evaluation or test-label parsing. Original pooling and ordinary MSE retained. Best checkpoint selected by validation Overall MSE.

| Dataset | Arm | Overall RMSE | Cliff RMSE | Noncliff RMSE | Parameters | Epochs | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| CHEMBL234_Ki | full | 0.741869 | 0.771413 | 0.717966 | 6021198 | 32 | 177.9 |
| CHEMBL234_Ki | short | 0.722070 | 0.791204 | 0.663123 | 6020358 | 21 | 92.6 |
| CHEMBL234_Ki | cross | 0.713777 | 0.770784 | 0.665903 | 6811398 | 29 | 171.3 |
| CHEMBL244_Ki | full | 0.913696 | 1.038046 | 0.782840 | 6021198 | 63 | 330.2 |
| CHEMBL244_Ki | short | 0.913509 | 1.104363 | 0.694420 | 6020358 | 25 | 103.6 |
| CHEMBL244_Ki | cross | 0.924290 | 1.089244 | 0.741954 | 6811398 | 32 | 177.3 |

CHEMBL234_Ki: cross/full Cliff change = -0.08%; cross/short = -2.58%. Archived direct seed-42 Cliff RMSE = 0.670396.
Gate checks: {'cliff_vs_full': False, 'cliff_vs_short': False, 'overall_noninferiority': True, 'archived_direct_noninferiority': False}.

CHEMBL244_Ki: cross/full Cliff change = +4.93%; cross/short = -1.37%. Archived direct seed-42 Cliff RMSE = 0.861737.
Gate checks: {'cliff_vs_full': False, 'cliff_vs_short': False, 'overall_noninferiority': False, 'archived_direct_noninferiority': False}.

Decision: No-Go: stop expansion; no extra seeds/tasks/FPPool/loss changes.

Verification: 6 completed runs; saved metrics independently recomputed; best checkpoints replayed on cuda; maximum absolute replay error 8.8817842e-16; retained initialization and prediction head identical across arms.

Limits: one training seed and two datasets, capacity/normalization differ between arms, labels equal -log10(nM) on audited training rows (pKi is shifted by9); the earlier standardized-label description is corrected without changing metrics. This does not establish universal failure or an attention-specific causal effect.

Training code commit: `b67cc089ed567bb3d762dcf7312c5835f3cbc76d`. Dirty working tree at launch: `False`. Raw data, predictions and checkpoints remain ignored local artifacts.
