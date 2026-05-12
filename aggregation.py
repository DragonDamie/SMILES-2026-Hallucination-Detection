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

    # Default: last real token of the final transformer layer.
    layer = hidden_states[-1]          # (seq_len, hidden_dim)

    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())                 # scalar index

    feature = layer[last_pos]          # (hidden_dim,)

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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# Импортируем функции из вашего проекта
from model import get_model_and_tokenizer, MAX_LENGTH
from splitting import split_data

# Эта часть запустится, только если вы запустите этот файл как скрипт
if __name__ == '__main__':
    print("="*30)
    print("НАЧАЛО ПОИСКА ЛУЧШЕГО СЛОЯ")
    print("="*30)

    # --- 1. Загрузка данных ---
    import pandas as pd
    from tqdm import tqdm
    import torch

    DATA_FILE = "./data/dataset.csv"
    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])
    print(f"Загружено {len(all_labels)} примеров.")

    # --- 2. Загрузка модели ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    print(f"Модель загружена на {device}.")

    # --- 3. Узнаём количество слоёв ---
    with torch.no_grad():
        dummy = tokenizer("test", return_tensors="pt").to(device)
        out = model(**dummy, output_hidden_states=True)
        n_layers = len(out.hidden_states)
        hidden_dim = out.hidden_states[0].shape[-1]
    print(f"Модель имеет {n_layers} слоёв, размерность эмбеддингов: {hidden_dim}")

    # --- 4. Извлечение состояний последнего токена для каждого слоя ---
    # Создаём список для каждого слоя. В каждом списке будем хранить скрытые состояния для всех примеров.
    all_states_per_layer = [[] for _ in range(n_layers)]
    BATCH_SIZE = 4

    print("Начинаем извлечение скрытых состояний...")
    for start in tqdm(range(0, len(all_texts), BATCH_SIZE)):
        batch_texts = all_texts[start:start+BATCH_SIZE]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_tuple = outputs.hidden_states
        mask_cpu = attention_mask.cpu()

        for i in range(len(batch_texts)):
            # Находим индекс последнего реального токена в последовательности
            real_pos = (mask_cpu[i] == 1).nonzero(as_tuple=True)[0]
            # Если нет реальных токенов (маловероятно), берём первый
            last_token_idx = real_pos[-1].item() if len(real_pos) > 0 else 0

            for layer in range(n_layers):
                # Берём состояние [слой, пример, последний токен, все измерения]
                state = hidden_tuple[layer][i, last_token_idx, :].cpu().numpy()
                all_states_per_layer[layer].append(state)

    # Превращаем списки в матрици NumPy для удобства
    X_by_layer = [np.array(lst) for lst in all_states_per_layer]
    print("Извлечение завершено.")

    # --- 5. Оценка качества каждого слоя с помощью простого классификатора ---
    # Получаем разбиение на train/val. Возьмём первое (и единственное) разбиение.
    splits = split_data(all_labels, df, test_size=0.15, val_size=0.15)
    train_idx, val_idx, _ = splits[0]

    best_f1, best_layer = -1, 0
    layer_metrics = []

    print("Начинаем оценку слоёв...")
    for layer in range(n_layers):
        # Данные для этого слоя
        X_train, X_val = X_by_layer[layer][train_idx], X_by_layer[layer][val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        # Простой и быстрый классификатор
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        # Выбираем лучший F1
        if f1 > best_f1:
            best_f1, best_layer = f1, layer

        layer_metrics.append((layer, f1))
        print(f"Слой {layer:2d}: F1 = {f1:.4f}")

    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТ: Лучший слой — {best_layer} с F1 = {best_f1:.4f}")
    print("="*30)

    # --- 6. Строим график для наглядности ---
    layers = [m[0] for m in layer_metrics]
    f1_scores = [m[1] for m in layer_metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, f1_scores, marker='o', linestyle='-')
    plt.axvline(x=best_layer, color='r', linestyle='--', label=f'Лучший слой: {best_layer} (F1={best_f1:.3f})')
    plt.xlabel('Номер слоя')
    plt.ylabel('F1-мера на валидации')
    plt.title('Распределение сигнала галлюцинации по слоям модели')
    plt.legend()
    plt.grid(True)
    plt.savefig('layer_search.png')
    plt.show()
    print("График сохранён как 'layer_search.png'")
