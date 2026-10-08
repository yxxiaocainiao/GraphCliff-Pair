"""Reuse capture/replay, hashing retained tensors only after the full forward."""
import argparse
from pathlib import Path
import sys
import random
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import torch
from tools import trace_layer_forward as base


def plain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.replay, map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    train.set_seed(SimpleNamespace(seed=42))
    torch.use_deterministic_algorithms(True, warn_only=False)
    base.shared.global_max_pool = base.stable_max_pool
    model = base.previous.LayerSwap("retry_self").cuda().train()
    model.load_state_dict(saved["model"])
    q, r = saved["query"].cuda(), saved["reference"].cuda()
    hashes, values = [], {}
    for _ in range(50):
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        train.np.random.set_state(saved["numpy_rng"])
        result = model(q, r)
        digest = train.state_hash({"value": result})
        hashes.append(digest)
        values[digest] = result.detach().cpu().flatten().tolist()
        del result
    train.save_json(output / "plain.json", dict(capture_sha256=train.digest(args.replay),
                    repetitions=50, hashes=hashes, values=values, unique_predictions=len(values)))
    print("plain forward unique predictions:", len(values))


def observe(model):
    observed, handles, counts, stacks = {}, [], {}, {}
    for name, module in model.named_modules():
        if not name:
            continue
        def before(module, inputs, kwargs, name=name):
            count = counts.get(name, 0)
            counts[name] = count + 1
            key = f"{name}/{count}"
            stacks.setdefault(name, []).append(key)
            observed[key] = dict(rng_before=base.previous.full_rng(), training=module.training,
                                 inputs={str(i): x.detach() for i, x in enumerate(inputs) if isinstance(x, torch.Tensor)},
                                 non_tensor_inputs={str(i): repr(x) for i, x in enumerate(inputs) if not isinstance(x, torch.Tensor)},
                                 kwargs={k: x.detach() if isinstance(x, torch.Tensor) else repr(x) for k, x in kwargs.items()})
        def after(module, inputs, kwargs, result, name=name):
            key = stacks[name].pop()
            tensors = result if isinstance(result, tuple) else (result,)
            observed[key].update(rng_after=base.previous.full_rng(),
                                 outputs={str(i): x.detach() for i, x in enumerate(tensors) if isinstance(x, torch.Tensor)})
        handles.extend([module.register_forward_pre_hook(before, with_kwargs=True),
                        module.register_forward_hook(after, with_kwargs=True)])
    return observed, handles


def run(args):
    original_observe, original_save, original_sources = base.observe_forward, train.save_json, base.previous.sources
    def save(path, value):
        if Path(path).name == "forward485.json":
            for row in value["observed"].values():
                for field in ["inputs", "outputs", "kwargs"]:
                    row[field] = {key: base.describe(x) if isinstance(x, torch.Tensor) else x for key, x in row[field].items()}
            value["observation"] = "retain tensor references; CPU hashing after whole forward"
        original_save(path, value)
    def sources():
        return dict(original_sources(), **{str(Path(__file__).relative_to(train.ROOT)): train.digest(__file__)})
    base.observe_forward, train.save_json, base.previous.sources = observe, save, sources
    try:
        if args.check:
            base.check()
        elif args.plain:
            plain(args)
        elif args.replay:
            base.replay(args)
        else:
            base.capture(args)
    finally:
        base.observe_forward, train.save_json, base.previous.sources = original_observe, original_save, original_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--csv-root", default="D:/GraphCliff-main/benchmark_data")
    parser.add_argument("--replay")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()
    if not args.check and not args.output:
        parser.error("output is required")
    if args.plain and not args.replay:
        parser.error("plain mode requires replay capture")
    run(args)
