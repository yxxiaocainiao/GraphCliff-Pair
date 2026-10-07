"""Pure algebra for the conditional-weight draft; no data or model dependencies."""
from fractions import Fraction as F
import json
import math


def weights(p, q):
    assert sum(map(sum, p)) == sum(map(sum, q)) == 1
    pg, qg = [sum(row) for row in p], [sum(row) for row in q]
    assert all(x > 0 for x in pg + qg)
    joint = [[q[g][r] / p[g][r] for r in range(2)] for g in range(2)]
    marginal = [qg[g] / pg[g] for g in range(2)]
    conditional = [[joint[g][r] / marginal[g] for r in range(2)] for g in range(2)]
    return joint, marginal, conditional


def main():
    p = [[F(72, 100), F(18, 100)], [F(2, 100), F(8, 100)]]
    q = [[F(5, 100), F(5, 100)], [F(72, 100), F(18, 100)]]
    joint, marginal, conditional = weights(p, q)
    for g in range(2):
        assert sum(p[g][r] * conditional[g][r] for r in range(2)) == sum(p[g])
        for r in range(2):
            assert joint[g][r] == marginal[g] * conditional[g][r]
            assert p[g][r] * conditional[g][r] == sum(p[g]) * q[g][r] / sum(q[g])
    # Only G changes: conditional weights are one, but target fitting can worsen.
    p0 = [[F(45, 100), F(45, 100)], [F(5, 100), F(5, 100)]]
    q0 = [[F(5, 100), F(5, 100)], [F(45, 100), F(45, 100)]]
    _, _, c0 = weights(p0, q0)
    assert c0 == [[1, 1], [1, 1]]
    # e=0 at G=0 and e=3 at G=1. Restricted constant hypotheses 0 or 2.
    def loss(mass, score):
        return sum(sum(mass[g]) * (score - 3*g)**2 for g in range(2))
    assert loss(p0, 0) < loss(p0, 2)
    assert loss(q0, 0) > loss(q0, 2)
    # Equal G margins: conditional equals full joint weighting.
    q1 = [[F(18, 100), F(72, 100)], [F(8, 100), F(2, 100)]]
    j1, m1, c1 = weights(p, q1)
    assert m1 == [1, 1] and j1 == c1
    # One-center 2-D isotropic Gaussian factorizes; integrate R analytically.
    h, g, r = 0.7, 0.3, -0.2
    normal_g = math.exp(-g*g/(2*h*h)) / (math.sqrt(2*math.pi)*h)
    normal_r = math.exp(-r*r/(2*h*h)) / (math.sqrt(2*math.pi)*h)
    joint_density = math.exp(-(g*g+r*r)/(2*h*h)) / (2*math.pi*h*h)
    assert math.isclose(joint_density, normal_g*normal_r, rel_tol=1e-12)
    # Integral of joint / mismatched G density is not one at G=0.
    integral_wrong = (1 / (math.sqrt(2*math.pi)*h)) / (1 / math.sqrt(2*math.pi))
    assert math.isclose(integral_wrong, 1/h) and not math.isclose(integral_wrong, 1)
    print(json.dumps({'status': 'passed', 'checks': [
        'joint_marginal_conditional_factorization', 'generic_margin_preserved',
        'hybrid_distribution_identity', 'generic_only_shift_noop_and_counterexample',
        'same_generic_margin_reduces_to_joint', 'gaussian_marginal_same_bandwidth',
        'mismatched_bandwidth_not_normalized'], 'fits_loads_predictions': 0}, indent=2))


if __name__ == '__main__':
    main()
