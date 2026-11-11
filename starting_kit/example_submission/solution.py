import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster(X, dataset_id=None):
    """
    Минимальный пример решения.

    Важно:
    - dataset_id можно использовать для разных стратегий;
    - этот пример намеренно простой и не является хорошим финальным решением.
    """
    X = np.asarray(X, dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    # Пример dataset-specific ветвления.
    # Это НЕ подсказка о реальном числе кластеров.
    if dataset_id in {"ds00", "ds01", "ds02"}:
        candidate_ks = range(2, 5)
    elif dataset_id in {"ds03", "ds04", "ds05"}:
        candidate_ks = range(2, 6)
    else:
        candidate_ks = range(2, 7)

    best_score = -np.inf
    best_labels = None

    for k in candidate_ks:
        if k >= len(X):
            continue

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(X_scaled)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        return np.zeros(len(X), dtype=int)

    return best_labels
