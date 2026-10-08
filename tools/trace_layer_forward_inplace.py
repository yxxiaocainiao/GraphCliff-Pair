"""Move snapshot CPU transfers after the target forward as well as hashes."""
import argparse
import copy
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from graphcliff_pair import train
from tools import trace_layer_forward as base
from tools import trace_layer_forward_deferred as deferred


def run(args):
    previous = base.previous
    constructor, old_plan, original_sources = previous.LayerSwap, previous.PLAN, previous.sources
    output = Path(args.output)
    def sources():
        result = original_sources()
        for file in [Path(__file__), Path(base.__file__), Path(deferred.__file__)]:
            result[str(file.relative_to(train.ROOT))] = train.digest(file)
        return result
    def factory(*values, **kwargs):
        model = constructor(*values, **kwargs)
        state = dict(step=0)
        def before(module, inputs):
            if not module.training:
                return
            state["step"] += 1
            if state["step"] == 485:
                state["saved"] = dict(model={k: v.detach() for k, v in module.state_dict().items()},
                                      query=inputs[0], reference=inputs[1], cpu_rng=torch.get_rng_state(),
                                      cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                                      numpy_rng=train.np.random.get_state())
                state["rng_before"] = previous.full_rng()
                state["observed"], state["handles"] = deferred.observe(module)
        def after(module, inputs, result):
            if module.training and state["step"] == 485:
                for handle in state["handles"]:
                    handle.remove()
                saved = state["saved"]
                saved["model"] = {k: v.cpu() for k, v in saved["model"].items()}
                saved["query"], saved["reference"] = copy.copy(saved["query"]).cpu(), copy.copy(saved["reference"]).cpu()
                torch.save(saved, output / "step485.pt")
                observed = state["observed"]
                for row in observed.values():
                    for field in ["inputs", "outputs", "kwargs"]:
                        row[field] = {key: base.describe(x) if isinstance(x, torch.Tensor) else x for key, x in row[field].items()}
                train.save_json(output / "forward485.json", dict(observed=observed, weight_sha256=train.state_hash(saved["model"]),
                                rng_before=state["rng_before"], prediction=base.describe(result),
                                prediction_values=result.detach().cpu().flatten().tolist(),
                                capture_sha256=train.digest(output / "step485.pt"),
                                observation="defer snapshot CPU transfers and tensor hashes until after forward"))
        model.register_forward_pre_hook(before)
        model.register_forward_hook(after)
        return model
    previous.LayerSwap, previous.PLAN, previous.sources = factory, base.PLAN, sources
    try:
        previous.worker(args)
    finally:
        previous.LayerSwap, previous.PLAN, previous.sources = constructor, old_plan, original_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-root", default="D:/GraphCliff-main/benchmark_data")
    args = parser.parse_args()
    args.dataset, args.variant = "CHEMBL234_Ki", "retry_self"
    run(args)
