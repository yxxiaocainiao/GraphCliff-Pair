"""One real training batch, backward only, before the fixed eight-fit pilot."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry.model import SingleBaseline
from experiments.mechanism_retry.run import read_safe
from experiments.mechanism_retry.repro import stable_max_pool
from tools.run_self_anchor_pilot import CONFIG, PROTOCOL, SelfAnchorSwap, retry


def main():
    torch = train.torch
    torch.set_num_threads(2)
    cfg = json.loads(CONFIG.read_text())
    path = Path("D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv")
    frame, graphs, _, _, pairs, valid = read_safe(path, cfg["split_seed"])
    train.set_seed(SimpleNamespace(seed=42))
    baseline = SingleBaseline(38, 13, hidden_size=256, num_layers=3, dropout=0)
    base = {k: v.clone() for k, v in baseline.state_dict().items()}
    del baseline
    q, r, yq, yr, _ = next(train.batches(pairs, graphs, frame, 32, torch.device("cuda")))
    old_pool = shared.global_max_pool
    shared.global_max_pool = stable_max_pool
    records, reference = [], None
    try:
        for arm in cfg["arms"]:
            train.set_seed(SimpleNamespace(seed=42))
            torch.use_deterministic_algorithms(True, warn_only=False)
            variant = arm["variant"]
            model = (SelfAnchorSwap(anchor=variant == "retry_anchor") if variant in ("retry_anchor", "retry_gated_delta")
                     else retry.LayerSwap(variant)).cuda()
            model.load_shared(base)
            common = {k: v for k, v in model.state_dict().items() if k != "innovation_gain"}
            head = train.state_hash(model.head.state_dict())
            if arm["name"] == "self":
                reference = dict(common=train.state_hash(common), prediction=model.eval()(q, r).detach())
            if arm["name"] in ("anchor", "gated_delta"):
                assert train.state_hash(common) == reference["common"]
                if arm["name"] == "anchor":
                    torch.testing.assert_close(model.eval()(q, r), reference["prediction"], atol=0, rtol=0)
            model.train()
            train.set_seed(SimpleNamespace(seed=42))
            torch.use_deterministic_algorithms(True, warn_only=False)
            prediction = model(q, r)
            loss = (prediction - (yq-yr)).square().mean()
            loss.backward()
            assert torch.isfinite(prediction).all() and torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            gain = getattr(model, "innovation_gain", None)
            if gain is not None:
                assert float(gain.grad.abs().sum()) > 0
            records.append(dict(arm=arm["name"], parameters=sum(p.numel() for p in model.parameters()),
                                head_sha256=head, common_sha256=train.state_hash(common),
                                first_batch_loss=float(loss.detach()), gain_gradient= None if gain is None else gain.grad.detach().cpu().tolist()))
            del model, prediction, loss
    finally:
        shared.global_max_pool = old_pool
    assert len({x["head_sha256"] for x in records}) == 1
    assert len(pairs) == 2632 and len(valid) == 292
    files = [CONFIG, PROTOCOL, Path(__file__), train.ROOT / "tools/run_self_anchor_pilot.py",
             *sorted((train.ROOT / "experiments/mechanism_retry").glob("*.py")),
             *sorted((train.ROOT / "graphcliff_pair").rglob("*.py"))]
    result = dict(dataset="CHEMBL234_Ki", input_sha256=train.digest(path), train_count=len(pairs), valid_count=len(valid),
                  records=records, source_sha256={str(p.relative_to(train.ROOT)): train.digest(p) for p in files},
                  optimizer_steps=0, real_forward_backward_checks=4, test_evaluated=False,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    train.save_json(train.ROOT / "experiments/mechanism_retry/self_anchor_preflight_20261008.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
