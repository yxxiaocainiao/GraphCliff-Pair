"""Synthetic finite-bitset checks; no molecular data, features or model imports."""
from itertools import combinations, product
from math import isclose, isfinite
from statistics import mean
import json


def distance(a, b):
    return 1 - len(a & b) / len(a | b) if a | b else 0.


def relevance(a, b, c):
    if not all(isfinite(x) and 0 <= x <= 1 for x in (a, b, c)):
        raise ValueError('Finite binary-Jaccard distances in [0,1] required')
    if abs(a - b) > c + 1e-12 or c > a + b + 1e-12:
        raise ValueError('Triangle inequality violated')
    return max(0., min(1., (c - abs(a - b)) / (a + b))) if a + b else 0.


def summarize(q, refs, labels):
    if len(refs) < 2 or len(refs) != len(labels) or not all(map(isfinite, labels)):
        raise ValueError('Aligned finite reference labels, at least two witnesses')
    ds = [distance(q, x) for x in refs]
    weights, sali, radial = [], [], []
    for i, j in combinations(range(len(refs)), 2):
        weights.append(relevance(ds[i], ds[j], distance(refs[i], refs[j])))
        sali.append(abs(labels[i] - labels[j]) / max(distance(refs[i], refs[j]), 1e-3))
        radial.append((1 - ds[i]) * (1 - ds[j]))
    joint = mean(w * s for w, s in zip(weights, sali))
    separate = mean(weights) * mean(sali)
    return dict(joint=joint, separate=separate, alignment=joint-separate,
                sali=mean(sali), mean_weight=mean(weights),
                radial=mean(w*s for w,s in zip(radial,sali)))


if __name__ == '__main__':
    universe = [frozenset(i for i in range(3) if mask & (1 << i)) for mask in range(8)]
    for q, i, j in product(universe, repeat=3):
        a, b, c = distance(q, i), distance(q, j), distance(i, j)
        w = relevance(a, b, c)
        assert 0 <= w <= 1 and isclose(w, relevance(b, a, c), abs_tol=1e-12)
        if q == i or i == j:
            assert abs(w) <= 1e-12
    refs = [frozenset({i}) for i in range(3)]
    labels = [0., 0., 2.]
    qa, qb = frozenset({0, 1}), frozenset({0, 2})
    a, b = summarize(qa, refs, labels), summarize(qb, refs, labels)
    assert sorted(distance(qa, r) for r in refs) == sorted(distance(qb, r) for r in refs)
    assert isclose(a['joint'], 4/9) and isclose(b['joint'], 8/9)
    assert isclose(a['separate'], 20/27) and isclose(b['separate'], 20/27)
    assert isclose(a['radial'], 0.) and isclose(b['radial'], 1/6)
    assert a['sali'] == b['sali'] and a['mean_weight'] == b['mean_weight']
    chosen = min(range(2), key=lambda i: [a['joint'], b['joint']][i])
    assert chosen == 0
    assert [0., 9.][chosen] == 0 and [9., 0.][chosen] == 9
    for q in [qa, qb]:
        x = summarize(q, refs, labels)
        shifted = summarize(q, refs, [y + 7 for y in labels])
        scaled = summarize(q, refs, [-2*y for y in labels])
        assert 0 <= x['joint'] <= x['sali']
        assert isclose(x['joint'], shifted['joint'])
        assert isclose(scaled['joint'], 2*x['joint'])
        assert summarize(q, refs[::-1], labels[::-1]) == x
    # Collision blindness: identical fingerprints, different reference activities.
    collision = summarize(frozenset({1}), [frozenset({0})]*2, [0., 2.])
    assert collision['joint'] == 0 and collision['sali'] == 2000
    # At equal query distances, the SALI pair-distance amplification cancels.
    assert isclose(relevance(.5, .5, .2) * 2/.2,
                   relevance(.5, .5, .8) * 2/.8)
    for args in [(-.1, .2, .2), (.1, .1, .9), (float('nan'), .2, .2)]:
        try:
            relevance(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid distances accepted')
    print(json.dumps(dict(passed=True,bitset_triples=len(universe)**3,
                         qa=a,qb=b,collision=collision,real_fits=0,
                         model_loads=0,real_feature_recomputations=0)))
