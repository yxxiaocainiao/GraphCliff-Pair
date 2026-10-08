"""Thin six-epoch adapter: capture batch485 without editing the pinned trainer."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
import torch
from experiments.mechanism_retry.repro import stable_max_pool
from tools import check_layer_repro_long as previous

PLAN = train.ROOT / "experiments/mechanism_retry/repro_forward_screen.json"


def describe(tensor):
    return dict(sha256=train.state_hash({"value": tensor}), shape=list(tensor.shape),
                stride=list(tensor.stride()), dtype=str(tensor.dtype), storage_offset=tensor.storage_offset(),
                pointer_mod256=tensor.data_ptr() % 256)


def observe_forward(model):
    observed, handles, counts, stacks = {}, [], {}, {}
    for name, module in model.named_modules():
        if not name:
            continue
        def before(module, inputs, kwargs, name=name):
            count = counts.get(name, 0)
            counts[name] = count + 1
            key = f"{name}/{count}"
            stacks.setdefault(name, []).append(key)
            observed[key] = dict(rng_before=previous.full_rng(), training=module.training,
                                 inputs={str(i): describe(x) for i, x in enumerate(inputs) if isinstance(x, torch.Tensor)},
                                 non_tensor_inputs={str(i): repr(x) for i, x in enumerate(inputs) if not isinstance(x, torch.Tensor)},
                                 kwargs={k: describe(x) if isinstance(x, torch.Tensor) else repr(x) for k, x in kwargs.items()})
        def after(module, inputs, kwargs, result, name=name):
            key = stacks[name].pop()
            tensors = result if isinstance(result, tuple) else (result,)
            observed[key].update(rng_after=previous.full_rng(),
                                 outputs={str(i): describe(x) for i, x in enumerate(tensors) if isinstance(x, torch.Tensor)})
        handles.extend([module.register_forward_pre_hook(before, with_kwargs=True),
                        module.register_forward_hook(after, with_kwargs=True)])
    return observed, handles


def snapshot(model, q, r):
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                query=copy.copy(q).cpu(), reference=copy.copy(r).cpu(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=train.np.random.get_state())


def capture(args):
    plan = json.loads(PLAN.read_text())
    output = Path(args.output)
    constructor, old_plan, original_sources = previous.LayerSwap, previous.PLAN, previous.sources
    def sources():
        return dict(original_sources(), **{str(Path(__file__).relative_to(train.ROOT)): train.digest(__file__)})
    def factory(*values, **kwargs):
        model = constructor(*values, **kwargs)
        state = dict(step=0)
        def before(module, inputs):
            if not module.training:
                return
            state["step"] += 1
            if state["step"] != plan["capture_step"]:
                return
            before_rng = previous.full_rng()
            saved = snapshot(module, *inputs)
            assert previous.full_rng() == before_rng, "capture consumed RNG"
            torch.save(saved, output / "step485.pt")
            observed, handles = observe_forward(module)
            state.update(observed=observed, handles=handles,
                         weight_sha256=train.state_hash(saved["model"]), rng_before=before_rng)
        def after(module, inputs, result):
            if module.training and state["step"] == plan["capture_step"]:
                for handle in state["handles"]:
                    handle.remove()
                train.save_json(output / "forward485.json", dict(observed=state["observed"],
                                weight_sha256=state["weight_sha256"], rng_before=state["rng_before"],
                                prediction=describe(result), prediction_values=result.detach().cpu().flatten().tolist(),
                                capture_sha256=train.digest(output / "step485.pt")))
        model.register_forward_pre_hook(before)
        model.register_forward_hook(after)
        return model
    previous.LayerSwap, previous.PLAN, previous.sources = factory, PLAN, sources
    try:
        previous.worker(SimpleNamespace(output=args.output, dataset=plan["dataset"], variant=plan["variant"], csv_root=args.csv_root))
        assert (output / "forward485.json").exists()
    finally:
        previous.LayerSwap, previous.PLAN, previous.sources = constructor, old_plan, original_sources


def replay(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Trusted local diagnostic capture includes PyG Batch objects.
    saved = torch.load(args.replay, map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    train.set_seed(SimpleNamespace(seed=42))
    torch.use_deterministic_algorithms(True, warn_only=False)
    shared.global_max_pool = stable_max_pool
    model = previous.LayerSwap("retry_self").cuda().train()
    model.load_state_dict(saved["model"])
    q, r = saved["query"].cuda(), saved["reference"].cuda()
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    train.np.random.set_state(saved["numpy_rng"])
    observed, handles = observe_forward(model)
    try:
        result = model(q, r)
    finally:
        for handle in handles:
            handle.remove()
    train.save_json(output / "forward485.json", dict(observed=observed, weight_sha256=train.state_hash(model.state_dict()),
                    prediction=describe(result), prediction_values=result.detach().cpu().flatten().tolist(),
                    capture_sha256=train.digest(args.replay)))


def check():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Dropout(.2)).train()
    x = torch.ones(3, 4)
    state = torch.get_rng_state()
    expected = model(x)
    torch.set_rng_state(state)
    observed, handles = observe_forward(model)
    actual = model(x)
    for handle in handles:
        handle.remove()
    assert torch.equal(expected, actual)
    assert len(observed) == 3 and all("outputs" in row and "inputs" in row for row in observed.values())
    assert observed["2/0"]["rng_before"] != observed["2/0"]["rng_after"]
    from torch_geometric.data import Batch, Data
    graph = Batch.from_data_list([Data(x=x.clone())]).to("cuda" if torch.cuda.is_available() else "cpu")
    device = graph.x.device
    saved = snapshot(model, graph, graph)
    assert graph.x.device == device and saved["query"].x.device.type == "cpu"
    assert torch.equal(saved["query"].x, graph.x.cpu())
    print("forward observation preserves output and records dropout RNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--csv-root", default="D:/GraphCliff-main/benchmark_data")
    parser.add_argument("--replay")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif not args.output:
        parser.error("output is required")
    else:
        replay(args) if args.replay else capture(args)
