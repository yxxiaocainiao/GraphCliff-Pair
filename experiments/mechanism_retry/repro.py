"""Opt-in max-pool backward correction; historical runners stay pinned."""
import argparse
import graphcliff_pair.model as shared
from graphcliff_pair import train
from .context_scale import run as original_run


def stable_max_pool(x, batch, size=None):
    """Same maxima; native amax shares the gradient across exact ties."""
    if x.ndim != 2 or batch.ndim != 1 or len(batch) != len(x):
        raise ValueError("expected node features [N,H] and graph assignments [N]")
    size = (int(batch.max()) + 1 if len(batch) else 0) if size is None else size
    return x.new_zeros((size, x.size(1))).scatter_reduce_(
        0, batch[:, None].expand_as(x), x, reduce="amax", include_self=False)


def run(config, csv_root, output):
    original_pool, original_save = shared.global_max_pool, train.save_json
    def save(path, value):
        if path.name == "manifest.json":
            value["pooling_backward"] = "native scatter_reduce amax; share gradient at exact ties; historical torch_scatter route replaced"
        original_save(path, value)
    shared.global_max_pool, train.save_json = stable_max_pool, save
    try:
        original_run(config, csv_root, output)
    finally:
        shared.global_max_pool, train.save_json = original_pool, original_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.csv_root, args.output)
