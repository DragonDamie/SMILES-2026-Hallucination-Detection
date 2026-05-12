"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    
    # Если нет DataFrame или колонки для группировки — используем обычный стратифицированный сплит
    if df is None or "prompt" not in df.columns:
        # Обычное стратифицированное разбиение (как в исходном коде)
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        return [(idx_train, idx_val, idx_test)]
    
    # --- Групповое разбиение (если есть колонка "prompt") ---
    from sklearn.model_selection import GroupShuffleSplit
    
    groups = df["prompt"].values  # идентификаторы групп (например, текст промпта)
    
    # Шаг 1: разбиваем на (train+val) и test с учётом групп
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_val_idx, test_idx), = next(gss1.split(idx, y, groups))
    
    # Шаг 2: внутри train_val снова разбиваем на train и val с учётом групп
    # Для этого извлекаем метки и группы для train_val части
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]
    # Относительный размер валидации внутри train_val
    relative_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=random_state)
    # Нужно отобразить индексы из train_val обратно в глобальные индексы
    local_idx = np.arange(len(train_val_idx))
    (train_local, val_local), = next(gss2.split(local_idx, y_train_val, groups_train_val))
    
    # Преобразуем локальные индексы в глобальные
    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]
    
    return [(train_idx, val_idx, test_idx)]
