"""One runnable contract check plus 4 real-batch backward preflights, zero updates."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphcliff_pair import train
import graphcliff_pair.model as shared
from experiments.mechanism_retry.bond_order.model import BondLongPoly, BondOrderPair, PAIRS, attach_bond_kind
from experiments.mechanism_retry.model import LayerSwap
from experiments.mechanism_retry.repro import stable_max_pool
from experiments.mechanism_retry.run import read_safe
from tools.check_bond_commutator import operators


def check():
    torch, np = train.torch, train.np
    torch.set_num_threads(2)
    shared.global_max_pool = stable_max_pool
    records = []
    for device in ("cpu", "cuda"):
        train.set_seed(SimpleNamespace(seed=42))
        torch.use_deterministic_algorithms(True, warn_only=False)
        base = LayerSwap("retry_full", 32, 2).to(device).eval()
        models = [BondOrderPair(32, 2, mode=mode).to(device).eval() for mode in ("skew", "symmetric")]
        from graphcliff_pair.vendor.dataset_utils import smiles_to_graph
        from torch_geometric.data import Batch, Data
        graphs = []
        for s in ("CC=O", "c1ccccc1C", "[Na+]"):
            sample = smiles_to_graph(s, 0)
            graphs.append(attach_bond_kind(Data(x=sample.x, edge_index=sample.edge_index, edge_attr=sample.edge_attr), s))
        q = Batch.from_data_list(graphs).to(device)
        r = Batch.from_data_list(graphs[::-1]).to(device)
        with torch.no_grad():
            expected = base(q, r)
        zero_errors = []
        for model in models:
            loaded = model.load_state_dict(base.state_dict(), strict=False)
            assert len(loaded.missing_keys) == 2 and all(k.endswith("order_coeffs") for k in loaded.missing_keys) and not loaded.unexpected_keys
            with torch.no_grad():
                actual = model(q, r)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            zero_errors.append(float((actual-expected).abs().max()))
            with torch.no_grad():
                for layer in model.encoder.layers:
                    layer.long.order_coeffs.fill_(.1)
                torch.testing.assert_close(model(q, r), -model(r, q), atol=1e-6, rtol=1e-6)
            assert train.parameters(model)-train.parameters(base) == 80
        # Compare the sparse skew addition to the independently defined dense basis.
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC=O")
        ops = operators(mol)
        original = base.encoder.layers[0].long
        long = BondLongPoly(original).to(device).eval()
        with torch.no_grad():
            long.cheb_coeffs.zero_()
            long.order_coeffs[:, 0] = .3  # single/double path pair
        g = graphs[0].to(device)
        x = torch.randn(3, 32, device=device)
        from graphcliff_pair.vendor.model import normalize_edges
        weights = normalize_edges(3, g.edge_index, x.new_ones(g.edge_index.size(1)))
        # Independent CPU float64 oracle avoids comparing high-precision sparse
        # arithmetic against a GPU dense matmul that may use reduced mantissas.
        matrix = torch.tensor((ops[0]@ops[1]-ops[1]@ops[0])/2, dtype=torch.float64)
        dense = ((matrix @ x.detach().cpu().double())*.3).to(device=device, dtype=x.dtype)
        expected = long.activation(long.norm(dense))
        torch.testing.assert_close(long(x, g.edge_index, weights, g.bond_kind), expected, atol=2e-6, rtol=2e-6)
        print("CONTRACT_OK", device, flush=True)
        records.append(dict(device=device, zero_full_max_abs_difference=max(zero_errors),
                            zero_degradation_check=True, signed_swap_check=True, sparse_dense_check=True))
    for dataset in ("CHEMBL234_Ki", "CHEMBL244_Ki"):
        frame, graphs, _, _, tp, _ = read_safe(f"D:/GraphCliff-main/benchmark_data/{dataset}.csv", 42)
        needed = {p[k] for p in tp[:32] for k in ("query", "reference")}
        for row in needed:
            attach_bond_kind(graphs[row], frame.at[row, "smiles"])
        for mode in ("skew", "symmetric"):
            train.set_seed(SimpleNamespace(seed=42))
            torch.use_deterministic_algorithms(True, warn_only=False)
            model = BondOrderPair(mode=mode).cuda().train()
            assert train.parameters(model) == 6021318
            q, r, yq, yr, _ = next(train.batches(tp[:32], graphs, frame, 32, torch.device("cuda")))
            loss = torch.mean((model(q, r)-(yq-yr))**2)
            loss.backward()
            coeffs = [layer.long.order_coeffs for layer in model.encoder.layers]
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            gradients = [float(p.grad.norm()) for p in coeffs]
            assert all(x > 0 for x in gradients)
            records.append(dict(dataset=dataset, mode=mode, batch_size=32, parameters=train.parameters(model),
                                loss=float(loss.detach()), coefficient_gradient_norms=gradients))
            print("REAL_BATCH_OK", dataset, mode, gradients, flush=True)
            del model, q, r, loss
            torch.cuda.empty_cache()
    paths = [Path(__file__), train.ROOT / "experiments/mechanism_retry/bond_order/model.py"]
    train.save_json(train.ROOT / "experiments/mechanism_retry/bond_order_preflight_20261008.json",
                    dict(records=records, source_sha256={str(p.relative_to(train.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                         optimizer_steps=0, formal_training_runs=0, test_evaluated=False))


if __name__ == "__main__":
    check()
