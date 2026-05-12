"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    BEST_LAYER = 6
    layer = hidden_states[BEST_LAYER]          # (seq_len, hidden_dim)

    # Находим индекс последнего реального (не паддинг) токена
    real_positions = attention_mask.nonzero(as_tuple=False)
    if len(real_positions) == 0:
        last_token_idx = 0
    else:
        last_token_idx = real_positions[-1].item()

    # Возвращаем вектор этого токена
    return layer[last_token_idx]               # (hidden_dim,)

    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False)
    if len(real_positions) == 0: return torch.zeros(8)
    last_token_idx = real_positions[-1].item()
    # берём последний токен со всех слоёв
    layer_states = hidden_states[:, last_token_idx, :]
    n_layers = hidden_states.shape[0]
    # 1. мера ICR — норма разности между последовательными слоями
    icr_scores = [torch.norm(layer_states[i] - layer_states[i-1]).item() for i in range(1, n_layers)]
    # 2. LSD — косинусное сходство между слоями (диапазон [-1,1]; 1=полное сходство)
    cosine_sims = [torch.nn.functional.cosine_similarity(layer_states[i].unsqueeze(0), layer_states[i-1].unsqueeze(0)).item()
                   for i in range(1, n_layers)]
    # 3. запасные признаки: размах норм, среднеквадратичное отклонение, наклон
    norms = torch.norm(layer_states, dim=1)
    norm_range = (norms.max() - norms.min()).item()
    norm_std  = norms.std().item()
    norm_slope = (norms[-1] - norms[0]).item()
    # объединяем всё в вектор
    geo = torch.tensor(icr_scores + cosine_sims + [norm_range, norm_std, norm_slope], dtype=torch.float32)
    if geo.numel() == 0: geo = torch.zeros(8)
    return geo


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
