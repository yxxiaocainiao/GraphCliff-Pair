"""Pure rational toy check; no molecular data, fits or prediction calls."""
from fractions import Fraction as F
from itertools import combinations
import json


def summaries(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    pairs = list(combinations(values, 2))
    return mean, variance, sum(abs(a - b) for a, b in pairs) / len(pairs)


def support_distance(prediction, reference_labels):
    if not reference_labels:
        raise ValueError('Reference labels must not be empty')
    return min(abs(prediction - v) for v in reference_labels)


def check():
    a = list(map(F, [-5, -4, -4, -1, -1, 1, 1, 4, 4, 5]))
    b = [F(-5), F(-19, 5), F(-19, 5), F(-8, 5), F(-8, 5),
         F(8, 5), F(8, 5), F(19, 5), F(19, 5), F(5)]
    assert summaries(a) == summaries(b) == (F(0), F(59, 5), F(194, 45))
    ga, gb = support_distance(F(0), a), support_distance(F(0), b)
    assert (ga, gb) == (F(1), F(8, 5))
    assert support_distance(F(7), [v + 7 for v in b]) == gb
    assert support_distance(F(0), [-3 * v for v in b]) == 3 * gb
    # Same legal inputs and scores, opposite realized error rankings.
    assert (F(0) ** 2, F(4) ** 2) == (0, 16)
    assert (F(4) ** 2, F(0) ** 2) == (16, 0)
    assert support_distance(F(1), [F(1), F(1)]) == 0
    assert support_distance(F(10), [F(1), F(2)]) == 8
    try:
        support_distance(F(0), [])
    except ValueError:
        pass
    else:
        raise AssertionError('Empty reference must fail')
    return dict(status='passed',neighbors=10,equal_variance='59/5',
                equal_mean_pair_distance='194/45',support_distances=['1','8/5'],
                fits=0,prediction_calls=0,real_data_reads=0)


if __name__ == '__main__':
    print(json.dumps(check()))
