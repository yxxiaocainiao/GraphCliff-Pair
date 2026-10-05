# Direct GraphCliff + ACA-MSE: fixed pilot

New user authorization on 2026-10-05. Target is a meaningful research result toward CAS zone-2 publication, not a promise of acceptance. Old queues/goals remain paused. No test activity/cliff labels are parsed and no test inference is run.

## task_plan

1. Freeze upstream source, licensing, data selection and units before outcomes; adapt existing direct trainer using a pooled-embedding hook.
2. Verify upstream loss/MSE/gradient/empty-triplet contracts, original inference and initialization, then the six planned task/arm GPU smoke runs with checkpoint replay (plus retained excluded2835 preflight).
3. Publish the implementation/protocol milestone before full training. Run three tasks x seeds42/43/44 x MSE/ACA = maximum18 full-budget runs. No hyperparameter search or extra arms.
4. Independently audit predictions, actual checkpoints, best epoch, triplet counts, configuration and sources; publish all results and gate decision. No automatic 30-task expansion.

## Frozen choices and stopping rule

Tasks 234 and244 are retained failure/development tasks. The third is CHEMBL2047_EC50 (503 official training rows,195 cliff labels), the smallest training partition with at least20 cliff molecules in the carved validation set. Candidate2835 was rejected BEFORE full-budget training because it has only5 validation cliff molecules; its technical smoke runs are retained but excluded from performance evidence. The coverage rule uses label counts, not prediction errors. Selection uses official training metadata, not any test outcomes; no claim these tasks are unseen confirmatory data.

Original direct GraphCliff: hidden256/three layers/SAG/readout and identical initial weights; batch32,AdamW1e-4,wd1e-5,betas(.9,.95),clip1;100 maximum epochs,warmup10/cosine100,patience15. Same seeded sample order and validation-Overall-MSE checkpoint rule. Loss uses official ACALoss with squared=True,p=2,similarity_gate=False. Arms alpha=0 and alpha=0.1; alpha0.1 is the official class default, not a tuned optimum for GraphCliff. Lower/upper activity differences both1 log10 from the published unified window/default; training-only affine calibration maps thresholds into stored-label units. This is an ACA-MSE transfer, not exact replication of paper MAE settings. squared=True also squares embedding distances; we preserve and report this official behavior.

Budget is fixed before outcomes. Within this one 18-run pilot, stop immediately on nonfinite gradients/loss, changed data/source, failed checkpoint replay or absence of candidate training triplets in every batch. Efficacy gate after the fixed matrix: at least two of three tasks must have mean Cliff RMSE improvement >=3% vs matched alpha0, at least two of three seeds on each qualifying task improve, mean Overall RMSE on each task must not worsen >1%, and no task mean Cliff RMSE may worsen >3%. These are pragmatic exploration gates, not journal requirements. Failure means stop this transfer, no tuning,extra tasks,FPPool or30-task training. A pass only permits a separately justified next decision, not automatic expansion or novelty claims.

## License / provenance

ACA source is pinned to4c43e6024345d0ec4d40fdc88ad57224c898b6da (v3), GPLv3. Full loss source and upstream license are preserved verbatim; sources.json records byte hashes. New experiment adapters/tests are GPLv3; the combined ACA experiment is provided under GPLv3. Existing MIT core/vendor attribution and root license are retained. Runtime extracts and executes only the two upstream label-only definitions without changing their bodies; structural-gating paths are excluded from this pilot. This avoids importing optional structure-analysis dependencies. Reusing ACA is not an original contribution.

## notes

A training-only check shows current CSV y equals -log10(nM), with affine slope1 and tiny residual; earlier local reports calling these labels standardized are corrected using this evidence. No test labels are required for this check. Hooks capture the original pooled representation without replacing its encoder or prediction head. Calibration, triplet totals and zero-triplet batches are saved per run. alpha0 and alpha0.1 have identical model capacity and inference signatures.

## Paused after the first complete task

Six of18 runs completed and replayed. Task234 three-seed Cliff mean worsened3.01%, Overall4.55%; one seed improved and two worsened. Existing task-level deterioration veto triggered. User priority to pause on poor outcomes advanced assessment timing from the full matrix to the first complete task; thresholds unchanged. Remaining12 runs unfinished (one244 control interrupted,11 not started); no completion flag is created. See results.md, early_audit.json and stop_decision.json. Do not automatically resume or tune.
