"""M60 exact finite checks; no data/model/dependencies. Run with python selfcheck.py."""
from fractions import Fraction as F
import json


def expectation(mass, values):
    assert len(mass) == len(values) and sum(mass) == 1
    return sum(a * b for a, b in zip(mass, values))


def main():
    p, q = (F(9, 10), F(1, 10)), (F(1, 10), F(9, 10))
    w = tuple(b / a for a, b in zip(p, q))
    for loss in ((F(0), F(4)), (F(4), F(1))):
        assert expectation(p, tuple(a * b for a, b in zip(w, loss))) == expectation(q, loss)
    # Restricted hypotheses with e=(0,3): gA=(0,1), gB=(2,2).
    loss_a, loss_b = (F(0), F(4)), (F(4), F(1))
    assert expectation(p, loss_a) < expectation(p, loss_b)
    assert expectation(q, loss_a) > expectation(q, loss_b)
    # Identical marginals do not identify reversed conditional errors.
    same = (F(1, 2), F(1, 2))
    assert tuple(b / a for a, b in zip(same, same)) == (1, 1)
    source_errors, target_errors, accepted = (0, 3), (3, 0), 0
    assert source_errors[accepted] == 0 and target_errors[accepted] == 3
    # Two target worlds have identical source data but disagree off source support.
    support_p, support_q = (F(1), F(0)), same
    assert any(a == 0 and b > 0 for a, b in zip(support_p, support_q))
    assert (0, 0)[0] == (0, 9)[0] and (0, 0)[1] != (0, 9)[1]
    alpha = F(1, 2)
    relative = tuple(v / (1 - alpha + alpha * v) for v in w)
    assert all(0 <= v <= 1 / alpha for v in relative)
    mass = tuple(a * b for a, b in zip(p, relative))
    transported = tuple(v / sum(mass) for v in mass)
    assert transported == same and transported != q
    # Normalizing first then taking the nonlinear relative map changes the result.
    sample_mean = sum(w) / len(w)
    wrong_relative = tuple((v / sample_mean) / (1 - alpha + alpha * v / sample_mean) for v in w)
    assert wrong_relative != relative
    # At equal domain priors Bayes gives P(target|z)=q/(p+q).
    target_prob = tuple(b / (a + b) for a, b in zip(p, q))
    inverse = tuple((1 - v) / v for v in target_prob)
    assert inverse == tuple(a / b for a, b in zip(p, q)) and inverse != w
    print(json.dumps({"status": "passed", "checks": [
        "importance_identity", "restricted_class_benefit_not_ranking_guarantee",
        "conditional_shift_failure", "support_failure", "relative_bound_and_bias",
        "nonlinear_normalization_order", "discriminator_bayes_direction"
    ], "real_model_or_data_operations": 0}, indent=2))


if __name__ == "__main__":
    main()
