"""Synthetic algebra checks only; no model, data, training, or inference."""
from itertools import combinations, product
from math import isclose


def main():
    # Rows are synthetic queries, columns are paired reference scenarios.
    ordinary = [[0.2, 0.4, 0.1], [0.3, 0.2, 0.4], [0.8, 0.6, 0.9]]
    enhanced = [[0.25, 0.6, 0.2], [0.35, 0.22, 0.42], [0.7, 0.55, 0.85]]
    delta = [[t - a for t, a in zip(ts, aa)]
             for ts, aa in zip(enhanced, ordinary)]
    scenario = [[aa[0] + d for d in ds] for aa, ds in zip(ordinary, delta)]
    score = [max(row) for row in scenario]
    for aa, ts, ds, s in zip(ordinary, enhanced, delta, score):
        shifted = [(t + c) - (a + c)
                   for t, a, c in zip(ts, aa, [2.0, -1.0, 0.5])]
        assert all(isclose(d, v, abs_tol=1e-12) for d, v in zip(ds, shifted))
        assert s >= ts[0] - 1e-12
        assert s <= aa[0] + max(ts) - min(aa) + 1e-12
    constant_delta = [-0.1, -0.1, -0.1]
    assert isclose(0.4 + max(constant_delta), 0.3)

    # Exhaustively check the stated rectangular surrogate, including every m.
    for m in range(1, len(score) + 1):
        objectives = {}
        for accepted in combinations(range(len(score)), m):
            worst = max(sum(scenario[i][b] for i, b in zip(accepted, bs)) / m
                        for bs in product(range(3), repeat=m))
            assert isclose(worst, sum(score[i] for i in accepted) / m)
            objectives[accepted] = worst
        selected = tuple(sorted(sorted(range(len(score)), key=score.__getitem__)[:m]))
        assert isclose(objectives[selected], min(objectives.values()))

    # A counterexample: surrogate robustness can worsen actual accepted error.
    full = [0.2, 0.3]
    robust = [0.2 + max(0.0, 0.4), 0.3 + max(0.0, 0.0)]
    actual_error = [0.1, 0.5]
    before = min(range(2), key=full.__getitem__)
    after = min(range(2), key=robust.__getitem__)
    assert actual_error[after] > actual_error[before]
    print('PASS: cancellation, fallback, bounds, exhaustive surrogate optimum, harm counterexample; 0 models/data/fits.')


if __name__ == '__main__':
    main()
