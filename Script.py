import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from model import get_model_and_tokenizer, MAX_LENGTH
from splitting import split_data

if __name__ == '__main__':
    print("="*30)
    print("НАЧАЛО ПОИСКА ЛУЧШЕГО СЛОЯ")
    print("="*30)

    import pandas as pd
    from tqdm import tqdm
    import torch

    DATA_FILE = "./data/dataset.csv"
    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])
    print(f"Загружено {len(all_labels)} примеров.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    print(f"Модель загружена на {device}.")

    with torch.no_grad():
        dummy = tokenizer("test", return_tensors="pt").to(device)
        out = model(**dummy, output_hidden_states=True)
        n_layers = len(out.hidden_states)
        hidden_dim = out.hidden_states[0].shape[-1]
    print(f"Модель имеет {n_layers} слоёв, размерность эмбеддингов: {hidden_dim}")

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
            real_pos = (mask_cpu[i] == 1).nonzero(as_tuple=True)[0]
            last_token_idx = real_pos[-1].item() if len(real_pos) > 0 else 0

            for layer in range(n_layers):
                state = hidden_tuple[layer][i, last_token_idx, :].cpu().float().numpy()
                all_states_per_layer[layer].append(state)

    X_by_layer = [np.array(lst) for lst in all_states_per_layer]
    print("Извлечение завершено.")

    splits = split_data(all_labels, df, test_size=0.15, val_size=0.15)
    train_idx, val_idx, _ = splits[0]

    best_f1, best_layer = -1, 0
    layer_metrics = []

    print("Начинаем оценку слоёв...")
    for layer in range(n_layers):
        X_train, X_val = X_by_layer[layer][train_idx], X_by_layer[layer][val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1, best_layer = f1, layer

        layer_metrics.append((layer, f1))
        print(f"Слой {layer:2d}: F1 = {f1:.4f}")

    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТ: Лучший слой — {best_layer} с F1 = {best_f1:.4f}")
    print("="*30)

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
