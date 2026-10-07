"""Synthetic algebra and label-lineage checks only; no project data/model imports."""
from itertools import combinations
import json
from math import isclose
from statistics import mean, pvariance, variance


def energy(y, p):
    if len(y) != len(p) or len(y) < 2:
        raise ValueError("Aligned arrays with at least two witnesses required")
    return mean(((y[i] - y[j]) - (p[i] - p[j])) ** 2
                for i, j in combinations(range(len(y)), 2))


if __name__ == "__main__":
    cases = [([0., 2.], [0., 2.]), ([0., 2.], [2., 0.]),
             ([0., 2.], [-5., -3.]),
             (list(range(10)), [i * .7 + (-1) ** i for i in range(10)])]
    for y, p in cases:
        r = [a - b for a, b in zip(y, p)]
        assert isclose(energy(y, p), 2 * variance(r), abs_tol=1e-12)
        assert isclose(mean(v * v for v in r), pvariance(r) + mean(r) ** 2,
                       abs_tol=1e-12)
        assert isclose(energy(y, p), energy(y, [v + 7 for v in p]),
                       abs_tol=1e-12)
    assert energy(*cases[0]) == 0 and energy(*cases[1]) == 16
    assert energy(*cases[2]) == 0
    assert mean((a - b) ** 2 for a, b in zip(*cases[2])) == 25
    for y, p in [([], []), ([1.], [1.]), ([1., 2.], [1.])]:
        try:
            energy(y, p)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid arrays accepted")
    # Query q is in fold 0. Both neighbor OOF fits include q's label.
    folds = [{"q", "a"}, {"b", "c"}, {"d", "e"}]
    fit = set.union(*folds)
    q_reference = fit - folds[0]
    for i in q_reference:
        own_fold = next(f for f in folds if i in f)
        neighbor_training = fit - own_fold
        assert i not in neighbor_training and "q" in neighbor_training
    # Individual OOF predictions may be valid for themselves but depend on y_q.
    y_neighbors = [0., 2.]
    assert energy(y_neighbors, [0., 0.]) != energy(y_neighbors, [0., 1.])
    assert all("external_query" not in fit - f for f in folds)
    print(json.dumps(dict(passed=True, algebra_cases=len(cases),
                         invalid_cases=3, lineage_neighbors=len(q_reference),
                         constant_bias_mse=25, constant_bias_energy=0,
                         real_fits=0, model_loads=0, real_features=0)))
