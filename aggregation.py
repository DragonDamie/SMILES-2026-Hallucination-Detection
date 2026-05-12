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

    if layer_config is None:
        # Настройки по умолчанию для Qwen (можно подобрать через валидацию)
        layer_config = {
            'type': 'select_top_k',
            'layer_indices': None,  # Will be auto-selected
            'use_geometric': True
        }
    
    n_layers = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    
    # Find last real token position
    real_positions = attention_mask.nonzero(as_tuple=False)
    if len(real_positions) == 0:
        return torch.zeros(hidden_dim, device=hidden_states.device)
    last_token_idx = real_positions[-1].item()
    
    # Extract last token states from all layers
    layer_states = hidden_states[:, last_token_idx, :]  # (n_layers, hidden_dim)
    
    # Smart layer selection if indices not provided
    if layer_config['layer_indices'] is None:
        # Based on empirical studies, hallucination signals are strongest in mid-late layers
        # For Qwen with ~28-32 layers, this is around layers 14-22
        total_layers = n_layers
        start_layer = int(total_layers * 0.4)   # Start at 40% depth
        end_layer = int(total_layers * 0.7)     # End at 70% depth
        layer_config['layer_indices'] = list(range(start_layer, end_layer))
        print(f"Auto-selected layers {start_layer} to {end_layer} (total: {total_layers})")
    
    selected_states = layer_states[layer_config['layer_indices'], :]  # (k, hidden_dim)
    
    if layer_config['type'] == 'concatenate':
        # Concatenate all selected layers
        feature = selected_states.flatten()
    elif layer_config['type'] == 'select_top_k':
        # Option: select top-k layers based on variance (signal strength)
        # For now, just take last selected layer (often most informative)
        feature = selected_states[-1]  # Last selected layer
    elif layer_config['type'] == 'weighted_mean':
        # Learnable weights through feature extraction layers
        # This would require additional training parameters
        weights = torch.softmax(torch.ones(len(selected_states)), dim=0)
        feature = (selected_states * weights.view(-1, 1)).sum(dim=0)
    else:
        feature = selected_states[-1]  # Default: last selected layer
    
    return feature
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

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


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
