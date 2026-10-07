"""Isolate max-pool backward on the first divergent real training batch."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
import torch
import torch_scatter
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.repro import stable_max_pool as native_max


def observe_modules(model):
    observed, counts, handles = {}, {}, []
    def hook(name):
        def record(module, inputs, result):
            count = counts.get(name, 0)
            counts[name] = count + 1
            for i, tensor in enumerate(inputs):
                if isinstance(tensor, torch.Tensor):
                    key = f"{name}/{count}/input/{i}"
                    observed[key] = dict(forward=train.state_hash({"value": tensor}))
                    if tensor.requires_grad:
                        tensor.register_hook(lambda grad, key=key: observed[key].update(backward=train.state_hash({"grad": grad})))
            for i, tensor in enumerate(result if isinstance(result, tuple) else (result,)):
                if isinstance(tensor, torch.Tensor):
                    key = f"{name}/{count}/{i}"
                    observed[key] = dict(forward=train.state_hash({"value": tensor}))
                    if tensor.requires_grad:
                        tensor.register_hook(lambda grad, key=key: observed[key].update(backward=train.state_hash({"grad": grad})))
        return record
    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(hook(name)))
    return observed, handles


def run(capture, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    train.set_seed(SimpleNamespace(seed=42))
    torch.use_deterministic_algorithms(True)
    # This locally generated capture contains PyG Batch objects.
    saved = torch.load(capture, map_location="cpu", weights_only=False)
    model = LayerSwap("retry_centered").cuda().train()
    model.load_state_dict(saved["model"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    q, r = saved["query"].cuda(), saved["reference"].cuda()
    pools = []
    original = shared.global_max_pool
    def observe(x, batch, size=None):
        value = original(x, batch, size)
        item = dict(x=x.detach().clone(), batch=batch.detach().clone(), value=value.detach().clone())
        value.register_hook(lambda grad: item.update(upstream=grad.detach().clone()))
        pools.append(item)
        return value
    shared.global_max_pool = observe
    try:
        pred = model(q, r)
        trace = json.loads((Path(capture).parent / "trace.json").read_text())
        assert train.state_hash({"tensor": pred}) == trace["records"][-1]["prediction_sha256"], "captured forward must replay exactly"
        loss, _ = train.DeltaLoss("mse")(pred, saved["target"].cuda(), 1)
        loss.backward()
    finally:
        shared.global_max_pool = original
    results = []
    for side, item in zip(["query", "reference"], pools):
        x, batch, value, upstream = (item[k] for k in ["x", "batch", "value", "upstream"])
        tie_count = x.new_zeros(value.shape).scatter_add_(0, batch[:, None].expand_as(x), (x == value[batch]).to(x.dtype))
        row = dict(side=side, tied_channels=int((tie_count > 1).sum()),
                   nonzero_upstream_tied_channels=int(((tie_count > 1) & (upstream != 0)).sum()),
                   input_sha256=train.state_hash(dict(x=x, batch=batch, upstream=upstream)))
        for name, pool in [("installed", original), ("native", native_max)]:
            grads, values, indices = set(), set(), set()
            for _ in range(100):
                z = x.detach().clone().requires_grad_()
                y = pool(z, batch)
                assert torch.equal(y, value), "replacement must preserve forward values"
                y.backward(upstream)
                assert torch.isfinite(z.grad).all()
                grads.add(train.state_hash({"gradient": z.grad}))
                values.add(train.state_hash({"value": y}))
                if name == "installed":
                    _, argmax = torch_scatter.scatter_max(z.detach(), batch, dim=0)
                    indices.add(train.state_hash({"argmax": argmax}))
            row[name] = dict(repetitions=100, unique_gradient_hashes=len(grads),
                             unique_forward_hashes=len(values), unique_separate_argmax_hashes=len(indices))
        results.append(row)
    assert len(results) == 2
    # Test the full captured backward before assigning blame to pooling.
    replays = []
    for _ in range(3):
        observed, handles = observe_modules(model)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        try:
            pred = model(q, r)
            loss, _ = train.DeltaLoss("mse")(pred, saved["target"].cuda(), 1)
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        replays.append(observed)
    report = dict(capture_sha256=train.digest(capture), strict=True, torch_scatter=torch_scatter.__version__,
                  results=results, full_backward_replays=replays, test_evaluated=False)
    train.save_json(output, report)
    print(json.dumps(dict(results=results, differing_modules=[key for key in replays[0]
                     if any(item[key] != replays[0][key] for item in replays[1:])]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.capture, args.output)
