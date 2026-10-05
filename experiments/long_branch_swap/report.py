"""Apply the preregistered gate; publish aggregates only."""
import argparse
from pathlib import Path
from .audit import read
from .run import ROOT
from graphcliff_pair import data, train

def report(output):
    output=Path(output)
    audited=read(output/"audit.json")
    manifest=read(output/"manifest.json")
    if manifest["config"]["seeds"]!=[42] or "limit_train_queries" in manifest["config"]:
        raise ValueError("screen gate requires full seed-42 screen")
    lines=["# LongPoly replacement: seed-42 screen", "", "Two development datasets; no test evaluation or test-label parsing. Original pooling and ordinary MSE retained. Best checkpoint selected by validation Overall MSE.", "", "| Dataset | Arm | Overall RMSE | Cliff RMSE | Noncliff RMSE | Parameters | Epochs | Seconds |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in audited["rows"]:
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['overall_rmse']:.6f} | {row['cliff_rmse']:.6f} | {row['noncliff_rmse']:.6f} | {row['parameters']} | {row['epochs']} | {row['elapsed_seconds']:.1f} |")
    decisions=[]
    provenance={}
    for dataset in manifest["config"]["datasets"]:
        arms={r["arm"]:r for r in audited["rows"] if r["dataset"]==dataset}
        direct_path=ROOT/"artifacts/interaction_seed42_20261004"/dataset/"seed42/direct/summary.json"
        pair_path=direct_path.parents[2]/"pairs.json"
        if read(pair_path)["valid_pairs"]!=read(output/dataset/"pairs.json")["valid_pairs"]:
            raise ValueError("archived direct pair mismatch")
        direct=read(direct_path)
        provenance[dataset]=dict(summary_sha256=data.digest(direct_path),pairs_sha256=data.digest(pair_path),cliff_rmse=direct["cliff_rmse"])
        cross=arms["cross"]
        checks=dict(cliff_vs_full=cross["cliff_rmse"]<=.95*arms["full"]["cliff_rmse"],cliff_vs_short=cross["cliff_rmse"]<=.95*arms["short"]["cliff_rmse"],overall_noninferiority=cross["overall_rmse"]<=1.01*min(arms[m]["overall_rmse"] for m in ["full","short"]),archived_direct_noninferiority=cross["cliff_rmse"]<=1.05*direct["cliff_rmse"])
        decisions.append(dict(dataset=dataset,checks=checks,go=all(checks.values())))
        lines.extend(["",f"{dataset}: cross/full Cliff change = {(cross['cliff_rmse']/arms['full']['cliff_rmse']-1)*100:+.2f}%; cross/short = {(cross['cliff_rmse']/arms['short']['cliff_rmse']-1)*100:+.2f}%. Archived direct seed-42 Cliff RMSE = {direct['cliff_rmse']:.6f}.",f"Gate checks: {checks}."])
    go=all(d["go"] for d in decisions)
    decision=dict(go=go,decisions=decisions,direct_provenance=provenance,scope="seed42 full-budget validation screen",test_evaluated=False)
    train.save_json(output/"decision.json",decision)
    lines.extend(["", "Decision: "+("Go: seeds 43/44 permitted by the frozen plan." if go else "No-Go: stop expansion; no extra seeds/tasks/FPPool/loss changes."), "", f"Verification: {audited['runs']} completed runs; saved metrics independently recomputed; best checkpoints replayed on {manifest['device']}; maximum absolute replay error {max(r['max_replay_error'] for r in audited['rows']):.8g}; retained initialization and prediction head identical across arms.", "", "Limits: one training seed and two datasets, capacity/normalization differ between arms, standardized activity rather than raw pKi. This does not establish universal failure or an attention-specific causal effect.", "", f"Training code commit: `{manifest['code_commit']}`. Dirty working tree at launch: `{manifest['code_dirty']}`. Raw data, predictions and checkpoints remain ignored local artifacts."])
    (output/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    print("GO" if go else "NO_GO")
    return go

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    report(p.parse_args().output)
