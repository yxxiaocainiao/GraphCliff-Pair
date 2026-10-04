import numpy as np
import pandas as pd

def split_train_valid(data: pd.DataFrame, valid_frac: float = 0.1, split_seed: int = 42):
    """Carve a validation set out of the official MoleculeACE training split.

    MoleculeACE ships only `train` / `test`. Selecting the checkpoint on `test`
    leaks the evaluation set, so we hold out a fraction of `train` as `valid`
    and use it for model selection / early stopping. The test split is never
    touched during training.

    The split is stratified on `cliff_mol` (when present) so that the cliff
    ratio of the training pool is preserved in both parts, and is fully
    determined by `split_seed`.

    Returns:
        (train_idx, valid_idx): positional indices into `data`.
    """
    train_pos = np.flatnonzero((data['split'] == 'train').to_numpy())
    if valid_frac <= 0:
        return train_pos, np.array([], dtype=int)

    if 'cliff_mol' in data.columns:
        strata = data['cliff_mol'].to_numpy()[train_pos]
    else:
        strata = np.zeros(len(train_pos), dtype=int)

    rng = np.random.RandomState(split_seed)
    train_idx, valid_idx = [], []
    for value in np.unique(strata):
        group = train_pos[strata == value]
        group = group[rng.permutation(len(group))]
        n_valid = int(round(len(group) * valid_frac))
        # Keep at least one molecule on each side of the split.
        n_valid = min(max(n_valid, 1), len(group) - 1) if len(group) > 1 else 0
        valid_idx.append(group[:n_valid])
        train_idx.append(group[n_valid:])

    train_idx = np.sort(np.concatenate(train_idx))
    valid_idx = np.sort(np.concatenate(valid_idx)) if valid_idx else np.array([], dtype=int)
    return train_idx, valid_idx
