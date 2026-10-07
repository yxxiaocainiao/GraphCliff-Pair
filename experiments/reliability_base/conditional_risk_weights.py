"""Label-free density weights on source-scaled evidence; no risk-model training.

Both frames must contain these seven columns, indexed by disjoint source_row IDs.
The caller must fit scaling on source only. Per-row diagnostics remain private.
"""
import numpy as np
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.neighbors import KernelDensity

FEATURES = ['prediction', 'nn_sim', 'local_dens', 'mol_size', 'rf_var', 'nbr_disp', 'sali_mean']


def normalized_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite log weights')
    weights = np.exp(values - values.max())
    if np.count_nonzero(weights) < 2:
        raise ValueError('Fewer than two numerically positive weights')
    return weights / weights.mean()


def joint_and_conditional_weights(source, target):
    """Four KDE fits shared by joint-IW and conditional-IW; no labels accepted."""
    arrays = []
    for frame in (source, target):
        if list(frame.columns) != FEATURES or len(frame) < 2:
            raise ValueError('Expected the exact seven evidence columns and at least two rows')
        if not frame.index.is_unique or any(not isinstance(i, (int, np.integer)) or i < 0 for i in frame.index):
            raise ValueError('Expected unique nonnegative source_row IDs as index')
        if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) or is_bool_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError('Evidence columns must be real numeric values')
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite evidence')
        arrays.append(values)
    if not source.index.intersection(target.index).empty:
        raise ValueError('Source and target identities overlap')
    s, t = arrays
    # Reuse SKADA DensityReweight's underlying sklearn estimator directly.
    joint_s = KernelDensity(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(s)
    joint_t = KernelDensity(bandwidth='scott', kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(t)
    margin_s = KernelDensity(bandwidth=joint_s.bandwidth_, kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(s[:, :5])
    margin_t = KernelDensity(bandwidth=joint_t.bandwidth_, kernel='gaussian', metric='euclidean', atol=0, rtol=0).fit(t[:, :5])
    log_joint = joint_t.score_samples(s) - joint_s.score_samples(s)
    log_generic = margin_t.score_samples(s[:, :5]) - margin_s.score_samples(s[:, :5])
    log_conditional = log_joint - log_generic
    if not np.isfinite(log_generic).all():
        raise ValueError('Nonfinite generic density ratio')
    joint, conditional = normalized_weights(log_joint), normalized_weights(log_conditional)
    return dict(joint=joint, conditional=conditional, log_joint=log_joint,
                log_generic=log_generic, log_conditional=log_conditional,
                source_bandwidth=joint_s.bandwidth_, target_bandwidth=joint_t.bandwidth_,
                conditional_ess=float(conditional.sum() ** 2 / np.square(conditional).sum()),
                conditional_max_share=float(conditional.max() / conditional.sum()))
