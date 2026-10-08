"""Pure synthetic checks; no molecular input, fitted model, or dependency."""
from fractions import Fraction as F


def mean(xs):
    if not xs:
        raise ValueError('Empty reference contexts')
    return sum(map(F, xs)) / len(xs)


if __name__ == '__main__':
    scores, target = [F(0), F(1), F(5)], F(2)
    avg = mean(scores)
    assert mean([(t-target)**2 for t in scores]) - (avg-target)**2 == mean([(t-avg)**2 for t in scores])
    assert mean([3, 3, 3]) > mean([0, 0, 0])  # Worse query selected after averaging.
    errors, full_scores = [0, 2], [0, 2]
    context_scores = [mean([3, 3, 3]), mean([0, 0, 0])]
    assert errors[min(range(2), key=full_scores.__getitem__)]**2 == 0
    assert errors[min(range(2), key=context_scores.__getitem__)]**2 == 4
    h = lambda r: F(r >= 1)
    assert h(mean([0, 0, 3])) == 1
    assert mean([h(r) for r in [0, 0, 3]]) == F(1, 3)
    changed = [scores[0], scores[1]+3, scores[2]]
    assert mean(changed)-avg == F(3, 3)
    assert mean(list(reversed(scores))) == avg
    try:
        mean([])
    except ValueError:
        pass
    else:
        raise AssertionError('Empty contexts must fail')
    assert 2*1403+3*585 == 4561
    assert 3*(585+3*585) == 7020
    assert 1403+2806+2806 == 7015
    print('PASS: identity, harmful ranking, nonlinear averaging, influence, empty input, budgets; real fits=0')
